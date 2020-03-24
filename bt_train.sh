#cool part https://gist.github.com/vratiu/9780109i
RED='\033[0;31m'
Cyan='\033[0;36m'
Green='\033[0;32m'
Yellow='\033[0;33m'
NC='\033[0m' # No Color
ROOT=$HOME/pds/bert
export DATA_DIR=$HOME/pds/bert/data


printf "${Yellow}Setup bert model PATH...\n"
export BERT_Chinese_DIR="${ROOT}/model/chinese_L-12_H-768_A-12"
printf "${Cyan}done!!!\n"

printf "${Yellow}Setup python venv environment...\n"
source ${HOME}/pds/DataServerTest/venv/bin/activate
printf "${Cyan}done!!!\n"
printf "${Yellow}Great!!! ${Cyan} Now, everyone is happy! ${RED}(◕ ‿ ◕ )!${NC}\n"

printf "${Yellow}Start training...\n"
#bert-serving-start -model_dir ${PATHNAME} -num_worker=1

python run_classifier.py \
  --task_name=demo \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_Chinese_DIR/vocab.txt \
  --bert_config_file=$BERT_Chinese_DIR/bert_config.json \
  --init_checkpoint=$BERT_Chinese_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$DATA_DIR/Demo_output/
