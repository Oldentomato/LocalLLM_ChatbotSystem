from konlpy.tag import Mecab
import urllib.request
import pandas as pd
from tokenizers import BertWordPieceTokenizer, CharBPETokenizer, SentencePieceBPETokenizer
from transformers import BertTokenizer, PreTrainedTokenizerFast
import sentencepiece as spm
import os

class Custom_Tokenizer:
    def __init__(self, vocab_dir, token_dir, vocab_file_name, raw_data_url, save_raw_name, model_name):
        self.vocab_dir = f"./out/{vocab_dir}"
        self.token_dir = f"./out/{token_dir}"
        self.vocab_file_name = f"./out/{vocab_file_name}"
        self.raw_data_url = raw_data_url
        self.save_raw_name = save_raw_name
        self.model_name = model_name

    def __check(self):
        if len(os.listdir(self.token_dir)) > 0:
            return True
        else:
            return False

    def __model_save(self, tokenizer):
        if tokenizer != None:
            tokenizer.save(self.vocab_dir)

            if self.model_name == "transformers":
                #transformers형태로 저장
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_dir)
                tokenizer.save_pretrained(self.token_dir)
            elif self.model_name == "bert":
                #bert transformers형태로 저장
                bert_token = BertTokenizer(self.vocab_dir)
                bert_token.save_pretrained(self.token_dir)
            elif self.model_name == "sentencebpe":
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_dir)
                tokenizer.save_pretrained(self.token_dir)

    def __Set_Bert(self):
        tokenizer = BertWordPieceTokenizer()

        vocab_size = 30000
        limit_alphabet = 6000
        min_frequency = 5
        special_tokens = ["<PAD>", "[CLS]", "[UNK]","[SEP]","[MASK]"]

        tokenizer.train(
            files=self.vocab_file_name,
            vocab_size=vocab_size,
            limit_alphabet=limit_alphabet,
            special_tokens = special_tokens,
            min_frequency=min_frequency
        )

        return tokenizer

    def __Set_BPET(self):

        vocab_size = 30000
        tokenizer = CharBPETokenizer(lowercase=False, unicode_normalizer='nfc')
        special_tokens = ["<s>", "</s>", "<unk>","</s>"]

        tokenizer.train(
            files=self.vocab_file_name,
            vocab_size=vocab_size,
            special_tokens = special_tokens
        )
        return tokenizer

    def __Set_Word2Vec(self):
        pass

    def __Set_SentenceBPEPiece(self):
        vocab_size = 30000
        limit_alphabet = 6000
        min_frequency = 5
        # spm.SentencePieceTrainer.Train('--input=./out/prepare.mecab.txt --model_prefix=./out/sentencetoken --vocab_size=30000 --model_type=bpe')
        # special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>'}
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train(
            self.vocab_file_name,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=special_tokens
        )

        return tokenizer


    def __Set_SentencePiece(self):
        spm.SentencePieceTrainer.Train('--input=./out/prepare.mecab.txt --model_prefix=./out/sentencetoken/sentencetoken --vocab_size=30000 --model_type=bpe --max_sentence_length=9999')
        return None



    def __Data_Preprocessing(self):
        print("get raw data and preprocessing")
        urllib.request.urlretrieve(self.raw_data_url, filename=self.save_raw_name)
        data = pd.read_table(self.save_raw_name)
        #debug
        # print(data.iloc[46471])

        #한글과 공백을 제외하고 모두 제거
        data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
        data = data.replace('', pd.NA)

        #결측치 제거
        data = data.dropna(axis=0)

        tokenizer = Mecab()
        with open(self.vocab_file_name, "w", encoding="utf-8") as f:
            for sentence in data['document']:
                f.write(f'{tokenizer.morphs(sentence)}\n')
        
    def run(self):
        if self.__check():
            print("tokenizer file is detected")
        else:
            print("tokenizer file is not detected \n generating...")
            self.__Data_Preprocessing()
            if self.model_name == "bert":
                tokenizer = self.__Set_Bert()
            elif self.model_name == "bert":
                tokenizer = self.__Set_BPET()
            elif self.model_name == "sentencebpe":
                tokenizer = self.__Set_SentenceBPEPiece()
            elif self.model_name == "sentence":
                tokenizer = self.__Set_SentencePiece()

            self.__model_save(tokenizer)




    





