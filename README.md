pai -name tensorflow1120
    -Dscript='odps://voyager_algo_dev/resources/lzd_img_search_rank_train_v9.tar.gz'
    -DentryFile='schedule/pai_server_run.py'
    -Dtables="odps://voyager_algo/tables/image_search_fg_output_all_domain_recall_post_v8_shuffle/ds=${bizdate}/venture=${venture},odps://voyager_algo/tables/image_search_fg_output_all_domain_recall_post_v8_shuffle/ds=${test_date}/venture=${venture}"
    -DuserDefinedParameters="--restore_flag=0 --inter_op_parallelism_threads=16 --intra_op_parallelism_threads=64 --dataset_type=v2 --batch_size=512 --mode_run=train --num_epochs=1 --fg_conf=find_similar_rank/fg_seq.json --global_conf=find_similar_rank/global_conf_seq.json --model_name=find_similar.model_dann --save_checkpoint_secs=6000 --pai_file_name=pai_main --save_summaries_steps=1000 --test_params=dnn_l2_reg=0&dropout_rate=0.2"
    -Dcluster='{\"ps\":{\"count\":1,\"memory\":16000,\"cpu\":500},\"worker\":{\"count\":4,\"cpu\":300,\"gpu\":0,\"memory\":8000}}'
    -DcheckpointDir='oss://lazada-search-algo/search-guide/suntong/image_search/rank/all_domain/local_train/with_trans/${venture}/${train_date}/data/?role_arn=acs:ram::1823608139307835:role/service-identity.com.alibaba.pai.21557d53&host=oss-ap-southeast-1-internal.aliyuncs.com'
    -DautoEnablePsTaskFailover=false;
