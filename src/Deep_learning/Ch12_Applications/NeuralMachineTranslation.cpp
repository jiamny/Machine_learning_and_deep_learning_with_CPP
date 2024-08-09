/*
 * SpeechRecognition.cpp
 *
 *  Created on: Jul 20, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <cctype>
#include <set>
#include <map>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

namespace F = torch::nn::functional;

void plot_heatmap(torch::Tensor tsr, std::string xlab, std::string ylab, std::string tlt,
		std::vector<std::string> x_ticks, std::vector<std::string> y_ticks) {
	tsr = tsr.cpu().squeeze().to(torch::kDouble);
	int nrows = tsr.size(0), ncols = tsr.size(1);

	std::vector<std::vector<double>> C;
	for( int i = 0; i < nrows; i++ ) {
		std::vector<double> c;
		for( int j = 0; j < ncols; j++ ) {
			c.push_back(tsr[i][j].item<double>());
		}
		C.push_back(c);
	}

	auto h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	auto ax = h->nexttile();
	matplot::heatmap(ax, C);
	matplot::colorbar(ax);
    ax->x_axis().ticklabels(x_ticks);
    ax->y_axis().ticklabels(y_ticks);
    matplot::xlabel(ax, xlab);
    matplot::ylabel(ax, ylab);
    if( tlt.length() > 2 ) {
    	matplot::title(ax, tlt.c_str());
    } else {
    	matplot::title(ax, "heatmap");
    }
    matplot::show();
}

std::string normalize_string(std::string s) {
	// convert lowercase
	std::transform(std::begin(s), std::end(s), std::begin(s), [](const std::string::value_type &x) {
	        return std::tolower(x, std::locale());
	    });
	std::smatch matches;
	if( std::regex_search(s, matches, std::regex("\\,")) )
		s = std::regex_replace(s, std::regex("\\,"), std::string(" , "));
	if( std::regex_search(s, matches, std::regex("\\.")) )
		s = std::regex_replace(s, std::regex("\\."), std::string(" . "));
	if( std::regex_search(s, matches, std::regex("\\!")) )
		s = std::regex_replace(s, std::regex("\\!"), std::string(" ! "));
	if( std::regex_search(s, matches, std::regex("\\?")) )
		s = std::regex_replace(s, std::regex("\\?"), std::string(" ? "));

	//s = std::regex_replace(s, std::regex("[^a-zA-Z,.!?]+"), std::string(" "));
	s = std::regex_replace(s, std::regex("\\s+"), std::string(" "));
	s = strip(s);
    return s;
}

torch::Tensor prepare_sequence(std::vector<std::string> seq, std::map<std::string, int> to_index) {
    //idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    std::vector<int> idxs;
    for(auto& w : seq) {
    	if( to_index.find(w) != to_index.end() ) {
    		idxs.push_back(to_index[w]);
    	} else {
    		idxs.push_back(to_index["<UNK>"]);
    	}
    }

    torch::Tensor t = torch::from_blob(idxs.data(), {static_cast<int64_t>(idxs.size()), 1},
			at::TensorOptions(torch::kInt32)).to(torch::kLong);
    return t.clone();
}

class Encoder : public torch::nn::Module {
public:
	Encoder(int _input_size, int _embedding_size, int _hidden_size, int _n_layers=1, bool bidirec=false) {

        input_size = _input_size;
        hidden_size = _hidden_size;
        n_layers = _n_layers;
        embedding_size = _embedding_size;

        embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(input_size, embedding_size));

        if(bidirec) {
            n_direction = 2;
            gru = torch::nn::GRU(torch::nn::GRUOptions(embedding_size, hidden_size)
            		.num_layers(n_layers)
            		.batch_first(true)
					.bidirectional(true));
        } else {
            n_direction = 1;
            gru = torch::nn::GRU(torch::nn::GRUOptions(embedding_size, hidden_size)
            		.num_layers(n_layers)
					.batch_first(true));
        }
        register_module("embedding", embedding);
        register_module("gru", gru);
    }

	torch::Tensor init_hidden(torch::Tensor inputs) {
		torch::Tensor hidden = torch::zeros({n_layers * n_direction, inputs.size(0), hidden_size}).to(inputs.device());
	    return hidden;
	}

	void init_weight(void) {
		torch::NoGradGuard nograd;
		embedding->weight = torch::nn::init::xavier_uniform_(embedding->weight);
		auto gru_state = gru->named_parameters(false);
		gru_state["weight_ih_l0"] = torch::nn::init::xavier_uniform_(gru_state["weight_ih_l0"]);
		gru_state["weight_hh_l0"] = torch::nn::init::xavier_uniform_(gru_state["weight_hh_l0"]);
	}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor inputs, torch::Tensor input_lengths) {
        // inputs : B, T (LongTensor)
        // input_lengths : real lengths of input batch (list)

        torch::Tensor hidden = init_hidden(inputs);
        torch::Tensor embedded = embedding->forward(inputs);

        torch::nn::utils::rnn::PackedSequence packed = torch::nn::utils::rnn::pack_padded_sequence(embedded, input_lengths, true, false);
        //packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        std::tuple<torch::nn::utils::rnn::PackedSequence, torch::Tensor> rt  = gru->forward_with_packed_input(packed, hidden);
        torch::nn::utils::rnn::PackedSequence outputs = std::get<0>(rt);
        hidden  = std::get<1>(rt);

        //batch_first = true
        torch::Tensor output_unpacked, output_lengths;
        std::tie(output_unpacked, output_lengths) = torch::nn::utils::rnn::pad_packed_sequence(outputs, true); // unpack (back to padded)

        if( n_layers > 1 ) {
            if( n_direction == 2 ) {
                hidden = hidden.index({Slice(-2, None), Slice(), Slice()}); //[-2:]
            } else {
                hidden = hidden[-1];
            }
        }

        std::vector<torch::Tensor> hs;
        for(int i = 0; i < hidden.size(0); i++ )
        	hs.push_back(hidden[i]);

        return std::make_tuple(output_unpacked, torch::cat(hs, 1).unsqueeze(1));
    }

private:
	int input_size = 0, embedding_size = 0, hidden_size = 0, n_layers = 0, n_direction = 0;
	torch::nn::Embedding embedding{nullptr};
	torch::nn::GRU gru{nullptr};

};

class Decoder : public torch::nn::Module {
public:
    Decoder(int input_size, int embedding_size, int _hidden_size, int _n_layers=1, float dropout_p=0.1) {

        hidden_size = _hidden_size;
        n_layers = _n_layers;

        // Define the layers
        embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(input_size, embedding_size));
        dropout = torch::nn::Dropout(torch::nn::DropoutOptions(dropout_p));

        gru = torch::nn::GRU(torch::nn::GRUOptions(embedding_size + hidden_size, hidden_size)
        		.num_layers(n_layers)
				.batch_first(true));

        linear = torch::nn::Linear(torch::nn::LinearOptions(hidden_size * 2, input_size));
        attn = torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size)); // Attention

        register_module("embedding", embedding);
        register_module("gru", gru);
        register_module("dropout", dropout);
        register_module("linear", linear);
        register_module("attn", attn);
    }

    torch::Tensor init_hidden(torch::Tensor inputs) {
    	torch::Tensor hidden = torch::zeros({n_layers, inputs.size(0), hidden_size}).to(inputs.device());
         return hidden;
    }

	void init_weight(void) {
		torch::NoGradGuard nograd;
		embedding->weight = torch::nn::init::xavier_uniform_(embedding->weight);
		auto gru_state = gru->named_parameters(false);
		gru_state["weight_ih_l0"] = torch::nn::init::xavier_uniform_(gru_state["weight_ih_l0"]);
		gru_state["weight_hh_l0"] = torch::nn::init::xavier_uniform_(gru_state["weight_hh_l0"]);
	    linear->weight = torch::nn::init::xavier_uniform_(linear->weight);
	    attn->weight = torch::nn::init::xavier_uniform_(attn->weight);
	}

    std::tuple<torch::Tensor, torch::Tensor> Attention(torch::Tensor hidden, torch::Tensor encoder_outputs, torch::Tensor encoder_maskings) {
        /*
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        */
    	//hidden = torch::unsqueeze(hidden[0], 2)
        hidden = hidden[0].unsqueeze(2);  // (1,B,D) -> (B,D,1)

        int batch_size = encoder_outputs.size(0);	// # B
        int max_len = encoder_outputs.size(1);		// # T
        torch::Tensor energies = attn->forward(encoder_outputs.contiguous().view({batch_size * max_len, -1})); // # B*T,D -> B*T,D
        energies = energies.view({batch_size,max_len, -1}); 			// # B,T,D
        torch::Tensor attn_energies = energies.bmm(hidden).squeeze(2);	// # B,T,D * B,D,1 --> B,T

//         if isinstance(encoder_maskings,torch.autograd.variable.Variable):
//             attn_energies = attn_energies.masked_fill(encoder_maskings,float('-inf'))#-1e12) # PAD masking

        torch::Tensor alpha = F::softmax(attn_energies,1);	// # B,T
        alpha = alpha.unsqueeze(1);							// # B,1,T
        torch::Tensor context = alpha.bmm(encoder_outputs); // # B,1,T * B,T,D => B,1,D

        return std::make_tuple(context, alpha);
    }

    torch::Tensor forward(torch::Tensor inputs, torch::Tensor context, int max_length,
    		torch::Tensor encoder_outputs, torch::Tensor encoder_maskings=torch::empty(0), bool is_training=false) {
        /*
        inputs : B,1 (LongTensor, START SYMBOL)
        context : B,1,D (FloatTensor, Last encoder hidden state)
        max_length : int, max length to decode # for batch
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        is_training : bool, this is because adapt dropout only training step.
        */
        // Get the embedding of the current input word
    	torch::Tensor embedded = embedding->forward(inputs);
    	torch::Tensor hidden = init_hidden(inputs);
        if( is_training )
            embedded = dropout->forward(embedded);

        std::vector<torch::Tensor> decode;
        // Apply GRU to the output so far
        for(auto& i : range(max_length, 0) ) {
        	torch::Tensor _;
            std::tie(_, hidden) = gru->forward(torch::cat({embedded, context}, 2), hidden);	// # h_t = f(h_{t-1},y_{t-1},c)
            torch::Tensor concated = torch::cat({hidden, context.transpose(0, 1)}, 2);		// # y_t = g(h_t,y_{t-1},c)
            torch::Tensor score = linear->forward(concated.squeeze(0));
            torch::Tensor softmaxed = F::log_softmax(score,1);
            decode.push_back(softmaxed);
            torch::Tensor decoded = std::get<1>(softmaxed.max(1));
            embedded = embedding->forward(decoded).unsqueeze(1);	// # y_{t-1}
            if( is_training )
                embedded = dropout->forward(embedded);

            // compute next context vector using attention
            torch::Tensor alpha;
            std::tie(context, alpha) = Attention(hidden, encoder_outputs, encoder_maskings);
        }
        //  column-wise concat, reshape!!
        torch::Tensor scores = torch::cat(decode, 1);
        return scores.view({inputs.size(0) * max_length, -1});
    }

    std::tuple<torch::Tensor, torch::Tensor> decode(torch::Tensor context, torch::Tensor encoder_outputs, std::map<std::string, int> target2index) {
    	torch::Tensor start_decode = torch::tensor({{target2index["<s>"]}}, torch::kLong).transpose(0, 1); //[[] * 1])).
		torch::Tensor embedded = embedding->forward(start_decode);
		torch::Tensor hidden = init_hidden(start_decode);

        std::vector<torch::Tensor> decodes, attentions;

        torch::Tensor decoded = embedded.clone();
        int dc = decoded.squeeze().data()[0].item<int>();
        while( dc != target2index["</s>"] ) { // # until </s>
        	torch::Tensor _;
            std::tie(_, hidden) = gru->forward(torch::cat({embedded, context}, 2), hidden);	// # h_t = f(h_{t-1},y_{t-1},c)
            torch::Tensor concated = torch::cat({hidden, context.transpose(0, 1)}, 2);		// # y_t = g(h_t,y_{t-1},c)
            torch::Tensor score = linear->forward(concated.squeeze(0));
            torch::Tensor softmaxed = F::log_softmax(score,1);
            decodes.push_back(softmaxed);
            decoded = std::get<1>(softmaxed.max(1)); //[1]
            dc = decoded.data().item<int>();
            embedded = embedding->forward(decoded).unsqueeze(1); // # y_{t-1}
            torch::Tensor alpha;
            std::tie(context, alpha) = Attention(hidden, encoder_outputs, torch::empty(0));
            attentions.push_back(alpha.squeeze(1));
        }
        return std::make_tuple(std::get<1>(torch::cat(decodes).max(1)), torch::cat(attentions));
    }

private:
    int hidden_size = 0, n_layers = 0;
    torch::nn::Embedding embedding{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Linear linear{nullptr}, attn{nullptr};
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pad_to_batch(std::vector<int> batch,
		std::vector<torch::Tensor> X_p, std::vector<torch::Tensor> y_p,
		std::map<std::string, int> source2index, std::map<std::string, int> target2index) {
	std::vector<torch::Tensor> Xs, ys; // = [item[0] for item in batch]
	std::vector<int> input_lens, target_lens;
    int max_x = 0, max_y = 0;
	for(auto& i : batch) {
		if( X_p[i].size(1) > max_x)
			max_x = X_p[i].size(1);

		if( y_p[i].size(1) > max_y)
			max_y = y_p[i].size(1);
		input_lens.push_back(X_p[i].size(1));
		target_lens.push_back(y_p[i].size(1));
	}


	for(auto& i : batch) {
		if( X_p[i].size(1) < max_x) {
			std::vector<int> pad;
			for(int j = 0; j < (max_x - X_p[i].size(1)); j++) {
				pad.push_back(source2index["<PAD"]);
			}
			torch::Tensor pads = torch::tensor(pad).view({1, -1}).to(torch::kLong);
			//std::cout << "source: " << pads.sizes() << " " << X_p[i].sizes() << '\n';
			Xs.push_back(torch::cat({X_p[i], pads}, 1));
		} else {
			Xs.push_back(X_p[i]);
		}

		if( y_p[i].size(1) < max_y) {
			std::vector<int> pad;
			for(int j = 0; j < (max_y - y_p[i].size(1)); j++) {
				pad.push_back(target2index["<PAD"]);
			}
			torch::Tensor pads = torch::tensor(pad).view({1, -1}).to(torch::kLong);
			//std::cout << "target: " << pads.sizes() << " " << X_p[i].sizes() << '\n';
			ys.push_back(torch::cat({y_p[i], pads}, 1));
		} else {
			ys.push_back(y_p[i]);
		}
	}

	torch::Tensor input_var = torch::cat(Xs);
	torch::Tensor target_var = torch::cat(ys);
	torch::Tensor input_len = torch::tensor(input_lens).to(torch::kLong);
	torch::Tensor target_len = torch::tensor(target_lens).to(torch::kLong);

	return std::make_tuple(input_var, target_var, input_len, target_len);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
	    std::cout << "CUDA available! Training on GPU." << std::endl;
	    device_type = torch::kCUDA;
	} else {
	    std::cout << "Training on CPU." << std::endl;
	    device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "load data\n";
	std::cout << "// -----------------------------------------------------------------\n";

	int MIN_LENGTH = 3;
	int	MAX_LENGTH = 25;
	int num_corpus = 30000;

	std::string file_name = "./data/fra.txt";
	std::string line;
	std::ifstream fL(file_name.c_str());
	std::vector<std::vector<std::string>> X_r;
	std::vector<std::vector<std::string>> y_r;

	int cnt = 0;
	if( fL.is_open() ) {

		while ( std::getline(fL, line) ) {
			cnt++;
			line = std::regex_replace(line, std::regex("\\\n"), "");
			line = strip(line);

			std::vector<std::string> strs = stringSplit(line, '\t');

			// trim space
			strs[0] = strip(strs[0]);
			strs[1] = strip(strs[1]);

		    if(strs[0] == "" || strs[1] == "")
		        continue;

		    std::vector<std::string> normalized_so = stringSplit(normalize_string(strs[0]), ' ');
		    std::vector<std::string> normalized_ta = stringSplit(normalize_string(strs[1]), ' ');

		    if(normalized_so.size() >= MIN_LENGTH && normalized_so.size() <= MAX_LENGTH
		       && normalized_ta.size() >= MIN_LENGTH && normalized_ta.size() <= MAX_LENGTH) {
		    	X_r.push_back(normalized_so);
		    	y_r.push_back(normalized_ta);
		    }

		    if( num_corpus > 0 && cnt >= num_corpus)
		    	break;
		}
	}
	fL.close();

	std::cout << "X_r: " << X_r.size() << '\n';
	std::cout << "y_r: " << y_r.size() << '\n';
	printVector(X_r[0]);
	printVector(y_r[0]);

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Build Vocab\n";
	std::cout << "// -----------------------------------------------------------------\n";
	std::set<std::string> uniX;
	std::vector<std::string> Xtks = flatten(X_r);
	for(std::string i : Xtks) {
		// finding position of i
		if( uniX.find(i) == uniX.end() )
			uniX.insert(i);
	}
	std::vector<std::string> source_vocab(uniX.begin(), uniX.end());

	std::set<std::string> uniy;
	std::vector<std::string> ytks = flatten(y_r);
	for(std::string i : ytks) {
		// finding position of i
		if( uniy.find(i) == uniy.end() )
			uniy.insert(i);
	}
	std::vector<std::string> target_vocab(uniy.begin(), uniy.end());

	std::cout << "source_vocab: " << source_vocab.size() << " target_vocab: " << target_vocab.size() << '\n';

	std::map<std::string, int> source2index;
	source2index["<PAD>"] = 0;
	source2index["<UNK>"] = 1;
	source2index["<s>"] = 2;
	source2index["</s>"] = 3;

	for(auto& vo : source_vocab ) {
		auto it = source2index.find(vo);
	    if( it ==  source2index.end() )
	        source2index[vo] = source2index.size();
	}

	std::map<int, std::string> index2source;
	for(auto& i : source2index) {
		index2source[i.second] = i.first;
	}


	std::map<std::string, int> target2index = {{"<PAD>", 0}, {"<UNK>", 1}, {"<s>", 2}, {"</s>", 3}};
	std::map<int, std::string> index2target;
	for(auto& vo : target_vocab) {
		auto it = target2index.find(vo);
	    if( it == target2index.end() )
	        target2index[vo] = target2index.size();
	}

	for(auto& i : target2index) {
		index2target[i.second] = i.first;
	}

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Prepare train data\n";
	std::cout << "// -----------------------------------------------------------------\n";
	std::vector<torch::Tensor> X_p, y_p;

	for(int i = 0; i < X_r.size(); i++) {
		std::vector<std::string> so = X_r[i];
		std::vector<std::string> ta = y_r[i];
		so.push_back("</s>");
		X_p.push_back(prepare_sequence(so, source2index).view({1, -1}));
		ta.push_back("</s>");
		y_p.push_back(prepare_sequence(ta, target2index).view({1, -1}));
	}

	torch::Tensor start_decode = torch::tensor({{target2index["<s>"]}}, torch::kLong).transpose(0, 1);
	std::cout << "start_decode: " << start_decode << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Configure Model and training\n";
	std::cout << "// -----------------------------------------------------------------\n";

	int EPOCH = 8,
	BATCH_SIZE = 64,
	EMBEDDING_SIZE = 300,
	HIDDEN_SIZE = 512;
	float LR = 0.001,
	DECODER_LEARNING_RATIO = 5.0;
	bool RESCHEDULED = false;
	Encoder encoder(source2index.size(), EMBEDDING_SIZE, HIDDEN_SIZE, 3, true);
	Decoder decoder(target2index.size(), EMBEDDING_SIZE, HIDDEN_SIZE * 2);
	encoder.to(device);
	decoder.to(device);
	encoder.train();
	decoder.train();
	encoder.init_weight();
	decoder.init_weight();

	/*
	auto a = torch::nn::GRU(torch::nn::GRUOptions(500, 50).num_layers(2));
	for(auto& p : a->named_parameters(false))
		std::cout << p.key() << "\n";
	*/

	auto loss_function = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().ignore_index(0)); //ignore_index=0

	torch::optim::Adam enc_optimizer = torch::optim::Adam(encoder.parameters(), LR);
	torch::optim::Adam dec_optimizer = torch::optim::Adam(decoder.parameters(), LR * DECODER_LEARNING_RATIO);

	bool prt = true;
	for(auto& epoch : range(EPOCH, 0) ) {

	    double losses = 0.;
	    std::list<std::vector<int>> batchs = data_index_iter(static_cast<int>(X_p.size()), BATCH_SIZE, true);
	    int i = 0;
	    for( auto& batch : batchs ) {
	    	i++;
	    	torch::Tensor inputs, targets, input_lengths, target_lengths;
	    	std::tie(inputs, targets, input_lengths, target_lengths) = pad_to_batch(
	    								batch, X_p, y_p, source2index, target2index);
	    	inputs = inputs.to(device);
	    	targets = targets.to(device);
	    	//target_lengths = target_lengths.to(device);
	    	//input_lengths = input_lengths.to(device);

			torch::Tensor input_masks = torch::eq(inputs, 0).to(torch::kUInt8).to(device);
			std::vector<int> st;
			for(auto& _ : range(static_cast<int>(targets.size(0)), 0)) {
				st.push_back(target2index["<s>"]);
			}
			torch::Tensor start_decode = torch::tensor(st, torch::kLong).unsqueeze(0).transpose(0, 1).to(device);

			if( prt ) {
	            std::cout << "start_decode: " << start_decode.sizes() << " targets.size(0): " << targets.size(0) << '\n';
	            prt = false;
			}
	        encoder.zero_grad();
	        decoder.zero_grad();
	        torch::Tensor output, hidden_c;
	        //std::cout << "inputs: " << inputs.sizes() << " input_lengths:\n" << input_lengths << '\n';
	        std::tie(output, hidden_c) = encoder.forward(inputs, input_lengths);
	        //std::cout << "output: " << output.sizes()
	        //		  << " hidden_c: " << hidden_c.sizes()
			//		  << " input_masks: " << input_masks.sizes() << '\n';

	        torch::Tensor preds = decoder.forward(start_decode, hidden_c, targets.size(1), output, input_masks, true);

	        auto loss = loss_function(preds, targets.view(-1));
	        losses += loss.data().item<double>();
	        loss.backward();
	        torch::nn::utils::clip_grad_norm_(encoder.parameters(), 50.0); // # gradient clipping
	        torch::nn::utils::clip_grad_norm_(decoder.parameters(), 50.0); // # gradient clipping
	        enc_optimizer.step();
	        dec_optimizer.step();

	        if( i % 200 == 0 ) {
	            printf("%02d/%d %03d/%d mean_loss : %0.2f\n", (epoch+1), EPOCH, i, static_cast<int>(batchs.size()), losses/200);
	            losses = 0.;
	        }
	    }

	    // You can use http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
	    if(RESCHEDULED == false && (epoch + 1)  == static_cast<int>(EPOCH/2.0)) {
	        LR *= 0.01;
	        //enc_optimizer = torch::optim::Adam(encoder.parameters(), LR);
	        //dec_optimizer = torch::optim::Adam(decoder.parameters(), LR * DECODER_LEARNING_RATIO);
	        torch::AutoGradMode enable_grad(true);
	        for(auto& param_group : enc_optimizer.param_groups() ) {
	        	param_group.options().set_lr(LR);
	        	std::cout << "enc_optimizer new lr: " << param_group.options().get_lr() << "\n";
	        }
	        for(auto& param_group : dec_optimizer.param_groups() ) {
	        	param_group.options().set_lr(LR * DECODER_LEARNING_RATIO);
	        	std::cout << "dec_optimizer new lr: " << param_group.options().get_lr() << "\n";
	        }
	        RESCHEDULED = true;
	    }
	}

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "Model testing and visualize attention\n";
	std::cout << "// -----------------------------------------------------------------\n";
	encoder.to(torch::kCPU);
	decoder.to(torch::kCPU);
	encoder.eval();
	decoder.eval();

	std::srand((unsigned) time(NULL)); // Set random number generator seed for RandT
	int sidx = RandT(0, static_cast<int>(X_p.size()));
	torch::Tensor input_ = X_p[sidx];
	torch::Tensor truth = y_p[sidx];
	torch::Tensor len = torch::tensor({static_cast<int>(input_.size(1))}).to(torch::kLong);

	torch::Tensor output, hidden;
	std::tie(output, hidden) = encoder.forward(input_, len);

	torch::Tensor pred, attn;
	std::tie(pred, attn) = decoder.decode(hidden, output, target2index);

	std::vector<std::string> x_ticks, y_ticks;
	std::vector<long> sX(input_.data_ptr<long>(), input_.data_ptr<long>() + input_.numel());
	std::vector<std::string> ss;
	for(auto& i : sX) {
		std::string s = index2source[i];
		if( s != "</s>")
			ss.push_back(s);
		x_ticks.push_back(s);
	}

	std::vector<long> tX(truth.data_ptr<long>(), truth.data_ptr<long>() + truth.numel());
	std::vector<std::string> ts;
	for(auto& i : tX) {
		if( i != 2 && i != 3) {
			std::string s = index2target[i];
			ts.push_back(s);
		}
	}

	std::vector<long> pX(pred.data_ptr<long>(), pred.data_ptr<long>() + pred.numel());
	std::vector<std::string> ps;
	for(auto& i : pX) {
		std::string s = index2target[i];
		if( s != "</s>")
			ps.push_back(s);
		y_ticks.push_back(s);
	}

	std::cout << "Source:     " << join(ss, " ") << '\n';
	std::cout << "Truth:      " << join(ts, " ") << '\n';
	std::cout << "Prediction: " << join(ps, " ") << '\n';

	plot_heatmap(attn, "Source", "Prediction", "", x_ticks, y_ticks);

	std::cout << "Done!\n";
	return 0;
}

