NAMES := predict train precision_bonus


all : 
	$(MAKE) $(NAMES)


predict predict/: bin_predict
train train/: bin_train
precision_bonus precision_bonus/: bin_precision_bonus

bin_train: train/src/main.rs shared_lib/src/lib.rs
	docker run -it -v $$(pwd):/app -w /app  rust cargo build --release --bin train
	@ln -sf target/release/train $@ 2> /dev/null
bin_predict: predict/src/main.rs shared_lib/src/lib.rs
	docker run -it -v $$(pwd):/app -w /app rust   cargo build --release --bin predict
	@ln -sf target/release/predict $@ 2> /dev/null
bin_precision_bonus: precision_bonus/src/main.rs shared_lib/src/lib.rs
	docker run -it -v $$(pwd):/app -w /app   rust cargo build --release --bin precision_bonus
	@ln -sf target/release/precision_bonus $@ 2> /dev/null

clean :
	rm -rf target/release/build/ 2>/dev/null
	$(shell rm data/{model.png,model.json})


fclean : clean
	@rm -rf $(addprefix bin_, $(NAMES)) target 2>/dev/null

re : fclean all

.PHONY: clean fclean
.IGNORE: clean fclean $(addprefix bin_, $(NAMES))
