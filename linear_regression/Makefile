NAMES := predict train precision_bonus

export RUST_FONTCONFIG_DLOPEN=1

export PATH := $(PWD)/.cargo/bin:$(PATH)

all : | .cargo
	@$(MAKE) $(NAMES)



.cargo :
	curl https://sh.rustup.rs -sSf | RUSTUP_HOME=./.rustup CARGO_HOME=./.cargo sh -s -- -y --no-modify-path --default-toolchain stable --profile minimal 2> /dev/null

predict predict/: bin_predict
train train/: bin_train
precision_bonus precision_bonus/: bin_precision_bonus

bin_train: train/src/main.rs shared_lib/src/lib.rs
	cargo build -r --bin train
	@ln -sf target/release/train $@ 2> /dev/null
bin_predict: predict/src/main.rs shared_lib/src/lib.rs
	cargo build -r --bin predict
	@ln -sf target/release/predict $@ 2> /dev/null
bin_precision_bonus: precision_bonus/src/main.rs shared_lib/src/lib.rs
	cargo build -r --bin precision_bonus
	@ln -sf target/release/precision_bonus $@ 2> /dev/null

clean :
	rm -rf target/release/build/ 2>/dev/null
	$(shell rm data/{model.png,model.json})


fclean : clean
	cargo clean
	@rm $(addprefix bin_, $(NAMES)) 2>/dev/null
	@read -p "delete cargo too ? y / [n] : " answer; \
	if [ "$$answer" == "y" ]; then \
		echo "deleting cargo..."; \
	    rm -rf ./.cargo ./.rustup; \
	fi

re : fclean all

.PHONY: clean fclean
.IGNORE: clean fclean $(addprefix bin_, $(NAMES)) .cargo
