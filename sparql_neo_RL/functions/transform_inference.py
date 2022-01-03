import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
CUDA = torch.cuda.is_available()

print(f"IS CUDA AVAILABLE: {CUDA}")

class TransformerInference:
    def __init__(
        self, model, tree_transform, pipeline_inverse, scalerx, maxcardinality
    ):
        self.tree_transform = tree_transform
        self.model = model
        if CUDA:
            self.model = self.model.cuda()
        self.pipeline_inverse = pipeline_inverse
        self.scalerx = scalerx
        self.maxcardinality = maxcardinality
        

    def pred2index_dict(self, x, pred_to_index, maxcardinality):
        resp = {}
        x = json.loads(x)
        for el in x.keys():
            if el in pred_to_index:
                resp[pred_to_index[el]] = float(x[el]) / maxcardinality
        return resp

    def prepare_query_level_data(
        self, x_test_query
    ):
        """
        Apply StandardScaller to columns except for json_cardinality that need other proccess
        """
        # Scale x_query data.
        xqtest = x_test_query.drop(columns=["json_cardinality"])
        x_test_scaled = self.scalerx.transform(xqtest)
        x_test_query = pd.concat(
            [
                pd.DataFrame(x_test_scaled, index=xqtest.index, columns=xqtest.columns),
                x_test_query[["json_cardinality"]],
            ],
            axis=1,
        )
        x_test_query["json_cardinality"] = x_test_query["json_cardinality"].apply(
            lambda x: self.pred2index_dict(
                x, self.tree_transform.get_pred_index(), self.maxcardinality
            )
        )

        return x_test_query

    
    def prepare_query_level_data_no_jc(
        self, x_test_query
    ):
        """
        Apply StandardScaller to columns except for json_cardinality that need other proccess
        """
        # Scale x_query data.
        xqtest = x_test_query.copy()
        x_test_scaled = self.scalerx.transform(xqtest)
        x_test_query = pd.DataFrame(x_test_scaled, index=xqtest.index, columns=xqtest.columns)
     

        return x_test_query
    
    
    def fix_tree(self, tree):
        """
        Trees in data must include in first position join type follow by predicates of childs. We check and fix this.
        """
        try:
            if len(tree) == 1:
                assert isinstance(tree[0], str)
                return tree
            else:
                assert len(tree) == 3
                assert isinstance(tree[0], str)
                preds = []
                if len(tree[0].split("ᶲ")) == 1:

                    tree_left = self.fix_tree(tree[1])
                    preds.extend(tree_left[0].split("ᶲ")[1:])

                    tree_right = self.fix_tree(tree[2])
                    preds.extend(tree_right[0].split("ᶲ")[1:])
                    preds = list(set(preds))
                    tree[0] = tree[0] + "ᶲ" + "ᶲ".join(preds)
                    return tree
                else:
                    return tree

        except Exception as ex:
            print(tree)
            return tree

    def json_loads(self, X, X_query):
        respX = []
        respX_query = []
        for x_tree, x_query in list(zip(X, X_query)):
            try:
                x_tree = json.loads(x_tree)
                respX.append(x_tree)
                respX_query.append(x_query)
            except:
                print("Error in data ignored!", x_tree, x_query)
        return respX, respX_query

    def index2sparse(self, tree, sizeindexes):
        resp = []
        for el in tree:
            if type(el[0]) == tuple:
                resp.append(self.index2sparse(el, sizeindexes))
            else:
                a = np.array(el)
                b = np.zeros((a.size, sizeindexes))
                b[np.arange(a.size), a] = 1
                onehot = np.sum(b, axis=0, keepdims=True)[0]
                resp.append(onehot)
        return tuple(resp)

    def collate_with_card(self, x):
        """
        Preprocess inputs values, transform index2vec values,
         them predict aec.encoder to dimensionality reduction
        """
        trees = []
        sizeindexes = len(self.tree_transform.get_pred_index())

        for tree, query_data in x:
            b = np.zeros((sizeindexes))
            try:
                for key in query_data[-1].keys():
                    b[key] = query_data[-1][key]
            except Exception as ex:
                print(ex)
                print(tree)
                print("Error en cardinalidades", str(query_data[-1]))
            trees.append(
                tuple(
                    [
                        self.index2sparse(tree, sizeindexes),
                        np.concatenate([query_data[:-1], b]).tolist(),
                    ]
                )
            )
        return trees

    def collate(self, x):
        #print("NeoRegression, collate method active")
        """Preprocess inputs values, transform index2vec values, them predict aec.encoder to dimensionality reduction"""
        trees = []
        targets = []
        sizeindexes = len(self.tree_transform.get_pred_index())
        for tree, query_data in x:
            trees.append(tuple([self.index2sparse(tree, sizeindexes), query_data]))
        return trees
    
    
    def prepare_with_card(self, x_test_tree, x_test_query):
        x_test_query = self.prepare_query_level_data(x_test_query)

        Xt, Xq = self.json_loads(x_test_tree, x_test_query.values)
        Xt = [self.fix_tree(x) for x in Xt]
        Xt = self.tree_transform.transform(Xt)
        pairs_val = list(zip(Xt, Xq))
        dataset_val = DataLoader(
            pairs_val,
            batch_size=64,
            num_workers=0,
            shuffle=False,
            collate_fn=self.collate_with_card,
        )
        return dataset_val
    
    def prepare(self, x_test_tree, x_test_query):
        x_test_query = self.prepare_query_level_data_no_jc(x_test_query)

        Xt, Xq = self.json_loads(x_test_tree, x_test_query.values)
        Xt = [self.fix_tree(x) for x in Xt]
        Xt = self.tree_transform.transform(Xt)
        pairs_val = list(zip(Xt, Xq))
        dataset_val = DataLoader(
            pairs_val,
            batch_size=64,
            num_workers=0,
            shuffle=False,
            collate_fn=self.collate,
        )
        return dataset_val

    def getpredictions_info(self, dataset_val):
        results = []
        results_extend = []
        for x in dataset_val:
            results_val = self.model(x)
            results.append(
                self.pipeline_inverse.inverse_transform(
                    results_val.cpu().detach().numpy()
                ).tolist()
            )
            results_extend.extend(
                self.pipeline_inverse.inverse_transform(
                    results_val.cpu().detach().numpy()
                ).tolist()
            )
        
        return results, results_extend
