import numpy as np

JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES

class SparqlTreeBuilder:
    def __init__(self, preds_map, preds_to_index):
        #print("SparqlTreeBuilder, __init__ method active")
        self.__preds_map = preds_map
        self.__preds_to_index = preds_to_index
        self.__index_to_preds = {index: pred for (pred, index) in preds_to_index.items()}
        self.lista_samples_aec = []

    def lista_samples_aec_ds(self):
        print("SparqlTreeBuilder, lista_samples_aec_ds method active")
        return self.lista_samples_aec

    def preds2onehot_tree(self, listaorg):
        #print("SparqlTreeBuilder, preds2onehot_tree method active")
        """
        Extraer dado un tree en forma de lista los predicados.
        """
        lista = listaorg.copy()
        for i in range(len(lista)):
            if type(lista[i]) == str:
                # Split by symbol to get list of tokens
                lista[i] = self.get_index_seq(lista[i])
            elif type(lista[i]) == list:
                lista[i] = self.preds2onehot_tree(lista[i])
        #print("------------------------------------------------------------------")
        return tuple(lista)

    def get_index_seq(self, cadena):
        #print("SparqlTreeBuilder, get_index_seq method active")
        row = []
        cadena_list = cadena.split("ᶲ")

        if cadena_list[0] not in self.__preds_to_index:
            print('OTHER_TPF',cadena_list, cadena_list[0])
            row.append(self.__preds_to_index['OTHER_TPF'])
        else:
            row.append(self.__preds_to_index[cadena_list[0]])
        for el in cadena_list[1:]:
            if el in self.__preds_to_index:
                row.append(self.__preds_to_index[el])
            else:
                print('OTHER_PRED',el)
                row.append(self.__preds_to_index['OTHER_PRED'])
        #         print(len(row))
        self.lista_samples_aec.append(row)

        return tuple(row, )

    def preds2onehot_from_list(self, cadena_list):
        print("SparqlTreeBuilder, preds2onehot_from_list method active")
        """
        Extraer dado un tree en forma de lista los predicados.
        """
        row = np.zeros(len(self.__preds_to_index))

        if cadena_list[0] not in self.__preds_to_index:
            row[self.__preds_to_index['OTHER_TPF']] = 1
        else:
            row[self.__preds_to_index[cadena_list[0]]] = 1
        for el in cadena_list[1:]:
            if el in self.__preds_to_index:
                row[self.__preds_to_index[el]] = 1
            else:
                row[self.__preds_to_index['OTHER_PRED']] = 1
        #         print(len(row))
        self.lista_samples_aec.append(row)
        return tuple(row, )

    def preds2onehot_from_str(self, cadena):
        print("SparqlTreeBuilder, preds2onehot_from_str method active")
        """
        Extraer dado un tree en forma de lista los predicados.
        """
        cadena_list = cadena.split("ᶲ")
        row = np.zeros(len(self.__preds_to_index))

        if cadena_list[0] not in self.__preds_to_index:
            row[self.__preds_to_index['OTHER_TPF']] = 1
        else:
            row[self.__preds_to_index[cadena_list[0]]] = 1
        for el in cadena_list[1:]:
            if el in self.__preds_to_index:
                row[self.__preds_to_index[el]] = 1
            else:
                row[self.__preds_to_index['OTHER_PRED']] = 1
        #         print(len(row))
        self.lista_samples_aec.append(row)
        return tuple(row, )

    def codificar_tree(self, tree):
        #print("SparqlTreeBuilder, codificar_tree method active")
        return self.preds2onehot_tree(tree)


class SPARQLTreeFeaturizer:
    def __init__(self):
        #print("SPARQLTreeFeaturizer, __init__ method active")
        self.__tree_builder = None
        self.__preds_map = {}
        self.__preds_to_index = {}

    def get_aec_ds(self):
        print("SPARQLTreeFeaturizer, get_aec_ds method active")
        return self.__tree_builder.lista_samples_aec_ds()

    def get_pred_index(self):
        #print("SPARQLTreeFeaturizer, get_pred_index method active")
        return self.__preds_to_index

    def get__preds_map(self):
        print("SPARQLTreeFeaturizer, get__preds_map method active")
        return self.__preds_map

    def fit(self, trees):
        #print("SPARQLTreeFeaturizer, fit method active")
        self.add_rest_indexes_features()
        self.extract_preds_index_map(trees)
        # stats_extractor = get_plan_stats(trees)
        self.__tree_builder = SparqlTreeBuilder(self.__preds_map, self.__preds_to_index)

    def fit_preds(self, train, val, test):
        #print("SPARQLTreeFeaturizer, fit_preds method active")
        self.add_rest_indexes_features()
        self.extract_preds_index_map(train)
        self.extract_preds_index_map(val)
        self.extract_preds_index_map(test)

        self.__tree_builder = SparqlTreeBuilder(self.__preds_map, self.__preds_to_index)

    def transform(self, trees):
        L = []
        for x in trees:
            L.append(self.__tree_builder.codificar_tree(x))
            
        #return [self.__tree_builder.codificar_tree(x) for x in trees]
        return L

    def transform_with_aec(self, ds_aec, aec):
        print("SPARQLTreeFeaturizer, transform_with_aec method active")
        aec.eval()
        newX = aec.encoder(ds_aec)
        return newX.cpu().detach().numpy()

    def num_operators(self):
        print("SPARQLTreeFeaturizer, num_operators method active")
        return len(ALL_TYPES)

    def get_one_hot_from_tuple(self, row_tuple):
        print("SPARQLTreeFeaturizer, get_one_hot_from_tuple method active")
        a = np.array(row_tuple)
        b = np.zeros((a.size, len(self.__preds_to_index)))
        b[np.arange(a.size), a] = 1
        onehot = np.sum(b, axis=0, keepdims=True)
        return onehot[0]
#         row = np.zeros(len(self.__preds_to_index))

#         for el in row_tuple:
#             row[el] = 1

#         return row

    ##########################################################
    def extract_preds_from_str(self, cadena):
        #print("SPARQLTreeFeaturizer, extract_preds_from_str method active")
        """
        Extraer dado un tree en forma de lista los predicados.
        """
        preds_map = {}
        cadena_list = cadena.split("ᶲ")
        for el in cadena_list[1:]:
            preds_map[el] = 1   
        return preds_map

    def extract_preds(self, lista):
        #print("SPARQLTreeFeaturizer, extract_preds method active")
        """
        Extraer dado un tree en forma de lista los predicados.
        """
        preds_map = {}
        for el in lista:
            if type(el) == str:
                upd = self.extract_preds_from_str(el)
                preds_map.update(upd)
            elif type(el) == list:
                upd = self.extract_preds(el)
                preds_map.update(upd)
        return preds_map

    def extract_preds_index_map(self, trees):
        #print("SPARQLTreeFeaturizer, extract_preds_index_map method active")
        """Extraer los mapas de predicados con su índice asociados para los vectores 1-hot"""
        try:
            for tree in trees:
                self.__preds_map.update(self.extract_preds(tree))
        except Exception as inst:
            print(inst)
        index = len(self.__preds_to_index)
        for key in list(self.__preds_map.keys()):
            self.__preds_to_index[key] = index
            index += 1

        #### SOLO PRINTEO  
        #print(type(self.__preds_to_index))
        for k,v in self.__preds_to_index.items():
            print(str(k) + "  :  " + str(v))
            print("----------------------------------------------------------------------------------------------")
        #### SOLO PRINTEO  
        
    def add_rest_indexes_features(self):
        #print("SPARQLTreeFeaturizer, add_rest_indexes_features method active")
        # Obtengo el último índice
        index = 0
        # Encode JOIN node with 1 and 0 a leaf node
        self.__preds_to_index["JOIN"] = index
        index += 1

        # Encode LEFT_JOIN node with 1 and 0 a leaf node
        self.__preds_to_index["LEFT_JOIN"] = index
        index += 1

        # Encode tpf types
        code_tpf = ['VAR_VAR_VAR','VAR_VAR_URI','VAR_URI_VAR', 'VAR_URI_URI', 'VAR_URI_LITERAL', 'VAR_VAR_LITERAL', 'URI_URI_LITERAL', 'URI_URI_VAR', 'URI_URI_URI', 'URI_VAR_VAR', 'URI_VAR_URI', 'URI_VAR_LITERAL',  'LITERAL_URI_VAR', 'LITERAL_URI_URI', 'LITERAL_URI_LITERAL','OTHER_TPF']
        for tpf in code_tpf:
            self.__preds_to_index[tpf] = index
            index += 1

        # Encode other preds as OTHER
        self.__preds_to_index['OTHER_PRED'] = index
        index += 1
