import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianModel()
    [BayesNet.add_node(i) for i in ["H","C", "M","B", "Q", "K", "D"]]
    BayesNet.add_edge("H","Q")
    BayesNet.add_edge("C","Q")
    BayesNet.add_edge("B","K")
    BayesNet.add_edge("M","K")
    BayesNet.add_edge("Q","D")
    BayesNet.add_edge("K","D")
    
    return BayesNet


def set_probability(bayes_net:BayesianModel):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    cpd_h = TabularCPD("H", 2, values= [[0.5], [0.5]])
    cpd_c = TabularCPD("C", 2, values= [[0.7], [0.3]])
    cpd_b = TabularCPD("B", 2, values= [[0.5], [0.5]])
    cpd_m = TabularCPD("M", 2, values= [[0.2], [0.8]])

    cpd_q = TabularCPD("Q", 2, values= [[0.95,	0.75,	0.45,	0.1], [0.05	, 0.25, 0.55, 0.9]], evidence=["H","C"], evidence_card=[2, 2])
    cpd_k = TabularCPD("K", 2, values= [[0.25	,0.05	,0.99	,0.85], [0.75	,0.95	,0.01	,0.15]], evidence=["B","M"], evidence_card=[2, 2])
    
    cpd_d = TabularCPD("D", 2, values= [[0.98	,0.4	,0.65	,0.01], [0.02	,0.6	,0.35	,0.99]], evidence=["K","Q"], evidence_card=[2, 2])

    bayes_net.add_cpds(cpd_h, cpd_c ,cpd_b ,cpd_m ,cpd_q ,cpd_k, cpd_d)
    return bayes_net


def get_marginal_double0(bayes_net: BayesianModel):
    """Calculate the marginal probability that Double-0 gets compromised.
    """
    agent = VariableElimination(bayes_net)
    double0_prob = agent.query(variables= ['D'], joint=False)['D'].values[1]
    return double0_prob


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    agent = VariableElimination(bayes_net)
    double0_prob = agent.query(variables= ['D'], evidence={"C":0},joint=False)['D'].values[1]
    return double0_prob


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    agent = VariableElimination(bayes_net)
    double0_prob = agent.query(variables= ['D'], evidence={"C":0, 'B': 1},joint=False)['D'].values[1]
    return double0_prob




def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    [BayesNet.add_node(i) for i in ["A","B","C","AvB","BvC","CvA"]]

    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("A", "AvB")

    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")

    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("C", "BvC")

    cpd_a = TabularCPD("A", 4, [[0.15] , [0.45], [0.30], [0.10]])
    cpd_b = TabularCPD("B", 4, [[0.15] , [0.45], [0.30], [0.10]])
    cpd_c = TabularCPD("C", 4, [[0.15] , [0.45], [0.30], [0.10]])

    

    a = build_tabels()
    values= [list(a.T[0])+ list(a.T[1])+ list(a.T[2])]
    cpd_ac = TabularCPD("CvA", 3, values , evidence=["A","C"], evidence_card=[4, 4])
    cpd_bc = TabularCPD("BvC", 3, values , evidence=["B","C"], evidence_card=[4, 4])
    cpd_ab = TabularCPD("AvB", 3, values , evidence=["A","B"], evidence_card=[4, 4])
    BayesNet.add_cpds(cpd_ab, cpd_ac, cpd_bc, cpd_a, cpd_b, cpd_c)

    return BayesNet

def build_tabels():
    skill_diff = {
    0:[0.10, 0.10, 0.80],
    1:[0.20, 0.60, 0.20],
    2:[0.15, 0.75, 0.10],
    3:[0.05, 0.90, 0.05],
    -1:[0.60, 0.20, 0.20],
    -2:[0.75, 0.15, 0.10],
    -3:[0.90, 0.05, 0.05]}
    a = zeros((16,3))
    b = zeros((4,4))
    col = 0
    for t_1 in [0, 1, 2, 3]: 
        for t_2 in [0, 1, 2, 3]:
            key = t_2 - t_1
            b[t_1][t_2] = key
            ab_0 = skill_diff[key][0]
            ab_1 = skill_diff[key][1]
            ab_2 = skill_diff[key][2]
            a[col] = [ab_0, ab_1, ab_2] 
            col += 1
    return a
def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    raise NotImplementedError
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)    
    # TODO: finish this function
    raise NotImplementedError
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    # TODO: finish this function
    raise NotImplementedError    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    raise NotImplementedError        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    raise NotImplementedError
