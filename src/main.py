import json, sys

from cross_valid import CrossValidation
from bagging import Bagging
from data_loader import DataLoader
from trainer import Trainer
from saver import Saver
from measures import Measures

#np.set_printoptions(threshold=sys.maxsize)
sys.path.append('.')

def evaluate_model_train_test(config):
    dataloader = DataLoader(config)
    train_ids, test_ids  = dataloader.train_test_split_from_ids(  test_ratio = .1)
    train_ids, valid_ids = dataloader.train_test_split_from_ids( train_ids, test_ratio = .111111111111)
    
    train_f  = dataloader.samples_from_ids( train_ids )
    test_f   = dataloader.samples_from_ids( test_ids  )
    valid_f  = dataloader.samples_from_ids( valid_ids  )

    x_train, y_train = dataloader.ml_prepare(train_f)
    x_test, y_test = dataloader.ml_prepare(test_f)
    x_valid, y_valid = dataloader.ml_prepare(valid_f)
        
    saver    = Saver(config)
    trainer  = Trainer(config, saver)
    model    = trainer.train(x_train, y_train, valid_data = (x_valid, y_valid))
    measures = Measures(model, x_test, y_test)
    measures.ROCCurve()
    measures = measures.eval()
    print(measures)
    
def evaluate_cross_valid(config) :
    CrossValidation(config)

def evaluate_bagging (config):
    bg = Bagging(config)
    #bg.generate_exp()
    #bg.evaluate()
    #bg.evaluate_ensemble()

if __name__ == "__main__":
    
    input_file  = sys.argv[1]
    input_file  = 'testing.json'

    with open(input_file, 'r') as f: 
        config = json.load(f)

        
    #valuate_model_train_test(config)
    evaluate_cross_valid(config)
    #evaluate_bagging(config)
    #test_data_loader(config)    
