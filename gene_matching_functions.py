from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.datasets import load_iris
# from sklearn import model_selection
import pandas as pd
import numpy as np
from scipy.spatial import distance
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pycpd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def set_seed(seed):
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


# ------------Data Loading--------------

def load_data_refer(fname="data\cegenelist_head_expression_and_space.csv"):
    data_refer = pd.DataFrame(pd.read_csv(fname))
    data_refer.index = data_refer['Neuron']
    data_refer.insert(0, '0', 0)
    data_refer.insert(0, '1', 1)
    data_refer = data_refer.T.drop_duplicates().T.drop(columns=['0', '1'])
    data_refer = data_refer.drop(columns=['Neuron'])
    return data_refer

def data_init(file_name, Neuron_all):
    dt = pd.DataFrame(pd.read_csv('data/' + file_name + '.csv'))
    dt = dt[dt['Neuron'].apply(lambda x: True if x in Neuron_all else False)]
    dt = dt.set_index('Neuron')

    return dt

def data_init_adjust(file_path, file_name, refer_data=None):

    if refer_data is None:
        refer_data = load_data_refer()
    NeuronNames = refer_data.index
    gene_cols = [gene for gene in refer_data.columns if gene not in ['Neuron','x','y','z']]

    dt = pd.DataFrame(pd.read_csv(file_path + file_name + '.csv'))
    dt = dt[dt['Neuron'].apply(lambda x: True if x in NeuronNames.values else False)]
    # dt.index = range(dt.shape[0])
    dt = dt.set_index('Neuron')

    dt, ss = rigid_adjust(dt, refer_data)
    if not(ss[0][0,0] > 0 and ss[0][1,1] > 0 and ss[0][2,2] > 0):
        print("Warning! Adjust in ", file_name)

    # gene_cols = refer_data.columns.tolist()[4:] # Neuron,x,y,z
    for i in dt.index:
        dt.loc[i, gene_cols] = refer_data.loc[i, gene_cols]

    return dt


def rigid_adjust(pts:pd.DataFrame, refer:pd.DataFrame, axis=[0,1,2]):
    src_pts = pts[['x','y','z']].values
    tgt_pts = refer[['x','y','z']].values.astype('float')
    reg = pycpd.AffineRegistration(X=tgt_pts[:,axis], Y=src_pts[:,axis])
    out_pts, ss = reg.register() # s,R,t
    out_data = pd.DataFrame(out_pts,columns=['x','y','z'])
    out_data.index = pts.index
    
    # out_data['z'] = pts['z']

    return out_data, ss



# -----------ID Matching-------------

def match_id(src:pd.DataFrame, tgt:pd.DataFrame, gene_list:list =None, xyz_weight:list =[1,1,1], maxtime01=1, maxtime10=2, w=0, fea_columns=None, gene_weight=6):
    """
    ### Inputs:
    Required:
    - source data : N1 * (3 + L + F), labeled
    - target data: N2 * (3 + L + F), unlabeled

    Optional:
    - gene_list: list of str with a length of L
    - xyz_weight: weights of three axes
    - maxtime01: maximum allowed times that a cell should express a gene but not detected, set as 1
    - maxtime01: maximum allowed times that a cell should not express a gene but detected, set as 2
    - gene_weight: penalty strength of a unallowed expression
    - w, fea_columns: other features and their weights

    ### Returns:
    - number of errors: int
    - errors (ori&pred) : list, the first line is the original labels, and the second line is the predicted labels
    - [pred_names, neuron_index] : predicted labels, and correspondent cell index from target data
    """
    if type(gene_list) in [set]:
        gene_list = list(gene_list)
    try:
        gene_exp_list = {}
        for i in src.index:
            exp = src.loc[i, gene_list].astype('int').values
            exp = ''.join(str(int(ee)) for ee in exp[0]) if exp.ndim == 2 else ''.join(str(int(ee)) for ee in exp)
            gene_exp_list[i] = exp
    except:
        gene_exp_list = None

    print(gene_list)
    print('begin')

    src_data = src.loc[:,['x','y','z']].values.astype('float')*xyz_weight; tgt_data = tgt.loc[:,['x','y','z']].values.astype('float')*xyz_weight
    src_name = src.index; tgt_name = np.array(tgt.index)
    src_fea = src[fea_columns].values.astype('float'); tgt_fea = tgt[fea_columns].values.astype('float')
    neuron_names = np.unique(src_name)

    N = tgt.shape[0]
    P = len(neuron_names)

    cost = np.zeros((N, P))

    for i in range(N):
        if gene_exp_list is not None:
            tgt_exp = tgt.iloc[i][gene_list].tolist()
            tgt_exp = ''.join(str(int(ee)) for ee in tgt_exp)
        tgt_xyz = tgt_data[i]
        
        for j, name in zip(range(P), neuron_names):
            if gene_exp_list is not None:

                indexes = np.where(src_name == name)
                distances = distance.cdist(tgt_xyz[None], src_data[indexes])
                cost[i,j] = np.mean(distances) + gene_weight * string_differ(tgt_exp, gene_exp_list[neuron_names[j]], maxtime01=maxtime01, maxtime10=maxtime10)


            else:
                indexes = np.where(src_name == name)
                distances = distance.cdist(tgt_xyz[None], src_data[indexes])
                cost[i,j] = np.mean(distances)

            if w > 0:
                for ii in indexes[0]:
                    cost[i,j] += w * np.linalg.norm(tgt_fea[i] - src_fea[ii])


    tgt_id, tgt_match_id = linear_sum_assignment(cost)
    tgt_name_match = neuron_names[tgt_match_id]
    tgt_name_with_match = tgt_name[tgt_id]
    errors = tgt_name_match != tgt_name_with_match
    errors = [tgt_name_with_match[errors].tolist(), tgt_name_match[errors].tolist()]
    # original labels, predicted labels

    return len(errors[0]), errors, [tgt_name_match, tgt_id]


def string_differ(s1:str, s2:str, maxtime01=1, maxtime10=2, weight10=2):
    """
    Positive difference from s1 to s2  
    s1 : unknown, experiment
    s2 : possible seq
    maxtime01 : s1=0, s2=1  
    maxtime10 : s1=1, s2=0
    Return: differ:True, Identical:False
    """
    assert len(s1) == len(s2)
    count01 = 0
    count10 = 0
    for w1, w2 in zip(s1, s2):
        if w1 == '0' and w2 == '1':
            count01 += 1
        elif w1 == '1' and w2 == '0':
            count10 += 1
    
    if count01 > maxtime01 or count10 > maxtime10:
        return count01 ** 2 + weight10 * count10 ** 2
    
    return 0


# ------------Add Noise------------
import copy


def add_noise(data:pd.DataFrame, expression_all:pd.DataFrame, gene_list='ALL', expression_threshold=3, infect_range=3, infect_range_z=0.5, seed=None, noise_prob=0.5):

    """
    Add poisson noises to the gene expression of a dataset. It's NOT very reasonable yet.
    - data : Neuron; x, y, z (, expression for each gene)
    - expression_all: The theoretic gene expression
    - gene_list: Genes that you want to add noises
    - infect_range: The maximum distance of a nearby cell
    - expression_threshold: A null expression of a gene in a certain cell might be infected if the expression number of its nearby cells > expression_threshold
    - noise_prob: [0,1], a bigger value means less noises
    ------
    Return
    ------
    - data_noisy : added noise in gene expression
    - data_org : data with theoretic gene expression
    - gene_change_count : list with length as gene_list
    - neuron_change_count : list with length as neuron numbers
    """
    if seed:
        set_seed(seed)

    xyz = data[['x','y','z']].values
    # xy = data[['x','y']].values
    # z = data[['z']].values
    if gene_list == 'ALL':
        gene_list = expression_all.columns.tolist()
        # gene_list.remove('Neuron')
    gene_change_count = np.zeros(len(gene_list))
    neuron_change_count = np.zeros(len(data))

    dist = distance.pdist(xyz)
    dist = distance.squareform(dist)
    # dist_z = distance.pdist(z)
    # dist_z = distance.squareform(dist_z)
    data_noisy = copy.deepcopy(data)
    neurons = data.index
    if gene_list[0] not in data_noisy.columns:
        for i in neurons:
            data_noisy.loc[i, gene_list] = expression_all.loc[i, gene_list]
    data_org = copy.deepcopy(data_noisy)
    
    for i, neuron in zip(range(len(neurons)), neurons):
        nn_points = np.where((dist[i,:] <= infect_range)) # & (dist_z[i,:] <= infect_range_z))
        nn_names = neurons[nn_points]
        nn_exp = []
        for name in nn_names:
            nn_exp.append(expression_all.loc[name, gene_list].values.tolist())
        nn_exp = np.array(nn_exp)
        
        for j in range(len(gene_list)):
            gene = gene_list[j]
            if data_noisy.loc[neuron, gene] == 0 and len(nn_exp)>1:
                lam = np.sum(nn_exp[:,j])
                if lam >= expression_threshold:
                    data_noisy.loc[neuron, gene] = np.random.random() > noise_prob
                    gene_change_count[j] += 1
                    neuron_change_count[i] += 1

    return data_noisy, data_org, gene_change_count, neuron_change_count

# ---------Plotting functions----------
def visualize_2d(*args, ax=None, axis=['x','y']):
    """View neurons in 2D space. Put any dataset(s) as input"""

    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.set_aspect(1)
    colors = 'cgrmy'
    i=0
    for x in args:
        if type(x) in [pd.DataFrame]:
            x = x[axis+[t for t in ['x','y','z'] if t not in axis]].values
        amin = np.min(x[...,2]); amax = np.max(x[...,2])
        for xx in x:
            ax.plot(xx[0], xx[1], '.', markersize=4, alpha=(xx[2]-amin)/(amax-amin)*0.8 + 0.2, c=colors[i])
        i += 1

def view_shift(data1:pd.DataFrame, data2:pd.DataFrame, axis=[0,1]):

    fig, ax = plt.subplots(1,1,dpi=200)

    for i in data1.index:
        x1 = data1.loc[i, ['x','y','z']].values
        try:
            x2 = data2.loc[i, ['x','y','z']].values
            ax.plot([x1[axis[0]],x2[axis[0]]], [x1[axis[1]], x2[axis[1]]], '-', linewidth=0.5)
            ax.plot(x1[axis[0]], x1[axis[1]], 'c.', markersize=4)
            ax.plot(x2[axis[0]], x2[axis[1]], 'r.', markersize=4)
        except:
            ax.plot(x1[axis[0]], x1[axis[1]], 'k.', markersize=6)

    ax.set_aspect(1)
    plt.show()


def plot_gene(gene, data:pd.DataFrame, expression, ax=None):
    """
    Plot cells that express a given gene    
    Args: gene name, data, expression data
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    visualize_2d(data.values, ax=ax)
    exp_neurons = expression.index[expression[gene] == 1]
    for n in exp_neurons:
        try:
            ax.plot(data.loc[n,'x'], data.loc[n,'y'], 'r.')
        except:
            continue
    plt.show()