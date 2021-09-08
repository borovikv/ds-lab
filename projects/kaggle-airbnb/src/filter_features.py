from tqdm import tqdm
from catboost import Pool, CatBoostClassifier
from sklearn.feature_selection import VarianceThreshold
import networkx as nx
import pandas as pd

def get_empty_columns(df, threshold=0.99, verbose=0):
    if verbose:
      print(f'Removing empty value columns')
    remove_cols = []
    for col in tqdm(list(df)):
        if df[col].isna().sum() / len(df) > threshold:
            remove_cols.append(col)
    return remove_cols
  
  
def get_low_variance_columns(df, verbose=0, threshold=0.0001):
    if verbose:
      print('started low variance analysis')
    numeric_cols = [col for col in list(df) if str(df[col].dtype) not in ('object', 'datetime64[ns]')]
    var_thr = VarianceThreshold(threshold=threshold)
    var_thr.fit(df[numeric_cols])
    low_variance_cols = [col for col, variance in zip(numeric_cols, var_thr.get_support()) if not variance]
    if verbose:
        print(f'low variante cols count for threshould {threshold}: {len(low_variance_cols)}')
        msg = '\n'.join(low_variance_cols)
    return low_variance_cols
  
  
def get_correlated_features_pairs(df, fi_weights=None, threshold=0.80, verbose=0, lower_bound=1000):
    if fi_weights is None:
        cols = list(df)
    else:
        cols = [col for col in list(df) if col in fi_weights]
    corr = abs(df[cols].corr())
    correlated = []
    cols = list(corr)
    if verbose:
      print('started collecting highly correlated column pairs')
    with tqdm(total=len(cols)) as pbar:
        for i, col in enumerate(cols):
            for j in range(i):
                if (corr.iloc[j][col] > threshold) and (~df[col].isna()).sum() > lower_bound and (~df[cols[j]].isna()).sum() > lower_bound:
                    correlated.append((cols[j], col, corr.iloc[j][col]))
            pbar.update(1)
    if verbose:
      print(f'total pairs found: {len(correlated)}')
    if correlated:
        cdf = pd.DataFrame(correlated)
        cdf.columns = ['col1', 'col2', 'coef']
        cdf.sort_values('coef', ascending=False, inplace=True)
        if verbose:
            print(f'showing head of correlated columns dataframe cdf:\n{cdf.head()}')
        return cdf
    else:
        if verbose:
            print('No correlated feautres found!')
        return None


def get_connected_components(cdf, verbose=0):
    if verbose:
        print('finding connected components')
    G = nx.Graph()
    edges = cdf[['col1', 'col2']].values.tolist()
    edges = [(el[0], el[1]) for el in edges]
    G.add_edges_from(edges)
    ccs = [list(i) for i in nx.connected_components(G)]
    if verbose:
        print(f'connected components found: {len(ccs)}')
    return ccs


def get_highly_correlated_removal_candidates(ccs, fi_weights, target='', verbose=0):
    # returns all highly correlated features
    if verbose:
        print('started selection of removal candidates:')
    if verbose:
        print(f'ccs and fi_weights sizes: {len(ccs), len(fi_weights)}')
    ccs_weighted = [[{el: fi_weights[el]} for el in cc] for cc in ccs]
    ccs_weighted = [sorted(el, key=lambda k: k[list(k.keys())[0]]) for el in ccs_weighted]
    msg = ''
    for ccw in ccs_weighted:
        msg = f"{msg}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        msg = f"{msg}" + '\n'.join([list(e.keys())[0] for e in ccw])
    if verbose:
        print(f'removing all but first in each group:\n{msg}')
    ccs_weighted = [el[:-1] for el in ccs_weighted]
    drop_features = [e for el in ccs_weighted for e in el]
    drop_features = [list(e.keys())[0] for e in drop_features]
    if target in drop_features:
        drop_features.remove(target)
    if verbose:
        print(f'selected for removal features count: {len(drop_features)}')
    return drop_features


def get_normalized_feature_weights(path, verbose=0):
    if verbose:
        print(f'Loading feature importance from:\n{path}')
    fi = pd.read_csv(path)
    if verbose:
        print(f'Loaded data of shape: {fi.shape}')
    fi_weights = fi[['feature_name', 'weight']].to_dict(orient='records')
    fi_weights = {el['feature_name']: el['weight'] for el in fi_weights}
    # normalizing weights
    weights_sum = sum([fi_weights[key] for key in fi_weights.keys()])
    fi_weights = {key: value / weights_sum for key, value in fi_weights.items()}

    return fi_weights


def generate_feature_importance_file(model_path, x_pickle_path, y_pickle_path, cat_features, save_to):
    x_test = pd.read_pickle(x_pickle_path)
    y_test = pd.read_pickle(y_pickle_path)

    model = CatBoostClassifier()
    model.load_model(model_path)
    test_pool = Pool(x_test, label=y_test,cat_features=cat_features)
    fi = model.get_feature_importance(test_pool)
    fi = pd.DataFrame({'feature_name': list(x_test), 'weight': fi})
    fi = fi.sort_values('weight', ascending=False).reset_index(drop=True)
    fi['w_sum'] = fi.weight.cumsum()
    fi.to_csv(save_to, index=False)

