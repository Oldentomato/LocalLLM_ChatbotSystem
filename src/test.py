from modules import Custom_Tokenizer


custom_token = Custom_Tokenizer(vocab_dir="sentencevocab/test.json", token_dir="sentencetoken/",
                                vocab_file_name="prepare.mecab.txt", raw_data_url="https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt",
                                save_raw_name="ratings.txt", model_name="sentence")


custom_token.run()