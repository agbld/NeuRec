#%%
import os
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool
import bottleneck as bn
from tqdm import tqdm
from esun.reco_utils import Evaluation

#%%
np.random.seed(2018)
random.seed(2018)
tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
if __name__ == "__main__":
    conf = Configurator("./esun/NeuRec.properties", default_section="hyperparameters")
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    # num_thread = int(conf["rec.number.thread"])

    # if Tool.get_available_gpus(gpu_id):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    dataset = Dataset(conf)

#%%
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    with tf.Session(config=config) as sess:
    # sess = tf.Session(config=config)
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)
            
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        
        # recommend
        to_cust_no = {v: k for k, v in dataset.userids.items()}
        to_wm_prod_code = {v: k for k, v in dataset.itemids.items()}
        
        recommendation = {}
        with tqdm(total=len(to_cust_no), desc='Recommend: ') as pbar:
            for k, v in to_cust_no.items():
                pred_r = model.predict([k], None)[0]
                top_5_item_id = bn.argpartition(-pred_r, 5)[:5]
                top_5_wm_prod_code = []
                for i in range(len(top_5_item_id)):
                    top_5_wm_prod_code.append(to_wm_prod_code[top_5_item_id[i]])
                recommendation[v] = top_5_wm_prod_code
                pbar.update(1)
        
        # evaluation
        evaluation = Evaluation('', conf['data.input.path'] + '/interaction_eval.csv', recommendation)
        score_all = evaluation.results()
        msg = "score: " + str(round(score_all, 4))
        model.logger.info(msg)