'''
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.

@author: Thomas Kelly
@author: Ona Igbinedion
@author: Raul Rodriguez
'''
import pandas as pd
import pomegranate as pm
import math
import itertools as it
import unittest


class AdEngine:

    def __init__(self, data_file, dec_vars, util_map):
        """
        Responsible for initializing the Decision Network of the
        AdEngine using the following inputs

        :param string data_file: path to csv file containing data on which
        the network's parameters are to be learned
        :param list dec_vars: list of string names of variables to be
        considered decision variables for the agent. Example:
          ["Ad1", "Ad2"]
        :param dict util_map: discrete, tabular, utility map whose keys
        are variables in network that are parents of a utility node, and
        values are dictionaries mapping that variable's values to a utility
        score, for example:
          {
            "X": {0: 20, 1: -10}
          }
        represents a utility node with single parent X whose value of 0
        has a utility score of 20, and value 1 has a utility score of -10
        """
        # TODO! You decide the attributes and initialization of the network!

        df = pd.read_csv(data_file)
        self.cols = df.columns
        self.bn = pm.BayesianNetwork.from_samples(
            X=df, state_names=self.cols, algorithm="exact")
        self.dec_vars = dec_vars
        self.util_map = util_map

        # Keeping track of values
        # df["col_name"].unique()
        self.vals = {}
        for e in self.cols:
            self.vals[e] = df[e].unique()

    # results returns as a large dictionary of the form
    # {Symbol1 : [(val1,prob),(val2,prob),(val3,prob)]}
    def symbolMatch(self, dist, col):
        result = {}
        for e in range(0, len(col)):
            if dist[e] == 1 or dist[e] == 0 or dist[e] == 2:
                result[col[e]] = dist[e]
            else:
                result[col[e]] = dist[e].items()
        return result

    def meu(self, evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, selects the ad content that maximizes expected utility
        and returns a dictionary over any decision variables and their
        best values plus the MEU from making this selection.

        :param dict evidence: dict mapping network variables to their
        observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: 2-Tuple of the format (a*, MEU) where:
          - a* = dict of format: {"DecVar1": val1, "DecVar2": val2, ...}
          - MEU = float representing the EU(a* | evidence)
        """
        # TODO: Implement the above!

        # okay so we figure out the values of the Dvars ... and then use combos from itertools? I think that would be a good idea.
        # so we want EU(D=0) = P(Y=y | D= 0) * U(Y=1)
        # so in code this looks like
        """
        We can create an array of tuples based on values of the dec_var
        [(0,we loop through each value of y(P(Y=y| D=0) * U(Y=y)) Do i need to figure out parents??
        (1, P(Y=y|D=1)*U(Y=y))
        and then we just find the max...
        lets see, so taking a first spin at this. get the arr of values from self.val using arr=self.val["D"] (will need to expand for multiple dec_vars)
        arr=[0,1]
        given util_var, I guess rn we will loop through and check which key exists... Make note of the key

        if it exists, put the resulting dict in a variable
        tuple_list = okay convert the dictionary to a list of tuples.
        for val in arr (will need to switch to combo list soon)
            for tup in  tuple_list
                
                result = helper_method(bn.predict_proba(dec_var: arr[0]),cols)
                tuples_result = result[noted_symbol]
                lets just assume for rn
                x =tuples_result[tup[0]][1] <-- this is the answer for y of P(D=0) (really P(Y=y |D=0) (saying y is 0)
                sum += x * tup[1]
                
        """
        # create a list of tuples with each of the dec_vars values
        list_tuple_dec_vars = []
        for var in self.dec_vars:
            values = self.vals[var]
            for v in values:
                list_tuple_dec_vars.append((var, v))
        # add in evidence varibles.
        evidence_count = 0
        if evidence != {}:
            for item in evidence.items():
                evidence_count += 1
                list_tuple_dec_vars.append(item)

        # now create a combonation list of evidence and dec_vars
        combo_list = it.combinations(
            list_tuple_dec_vars, len(self.dec_vars) + evidence_count)
        # so now we have a giant list of combos, note, some stuff is repeats
        # now create a list of dictionarys formatted with dec_var1 = val, dec_var2 = val, evidence =e
        combo_dict_list = []
        # item is the a list element
        for item in combo_list:
            dict_to_be_listed = {}
            # content is tuple ("symbol" :value)
            for content in item:
                # if symbol already in dictionary, then this is a combo with a repeat
                if content[0] in dict_to_be_listed:
                    break
                dict_to_be_listed[content[0]] = content[1]

            else:
                # if inner loop finished without a break, this means that the combo dict should be added to the list
                combo_dict_list.append(dict_to_be_listed)
                continue

        # we now have a list of dictionaries ready to be inserted into predict_proba.
        # now we loop through each
        # we also need to loop through values of the utility node and recover the value of this
        # we have the util map in format {Symbol : {val1 : util1 , val2 :util2}
        # to obtain the symbol in a brute force fashion, I will cycle through the cols until I find the key
        # ^ needs improvement
        symbol = ""
        for s in self.cols:
            if s in self.util_map:
                symbol = s
                break
        # we can also get the values we need to loop through for the util_node
        poss_util_vals = self.vals[symbol]
        # now we can easily call util_map in a loop with self.util_map[symbol][loop_val_it]
        # we also now which value to extract.
        # will attempt to store these results in a dictionary, though a list of tuples may be the only proper format...
        result_tuple_list = []
        for combo in combo_dict_list:
            sum = 0
            for val in poss_util_vals:
                # get resulting dictionary given combo dec vars and evidence. Symbol match is used to order the dictionary
                prob_dict = self.symbolMatch(
                    self.bn.predict_proba(combo), self.cols)
                # now we will get the tuple list from this dictionary.
                util_var_tuples = prob_dict[symbol]
                # now we must loop through these tuples until we find the value
                prob_answer = 0
                for tup in util_var_tuples:
                    if tup[0] == val:
                        prob_answer = tup[1]
                        break
                sum += prob_answer * self.util_map[symbol][val]
            result_tuple_list.append((combo.copy(), sum))
        # we have now stored the combo and sum in a list of tuples.
        # now loop through and find best.
        best_combo = {}
        best_util = 0
        for item in result_tuple_list:
            if item[1] > best_util:
                best_util = item[1]
                best_combo = item[0].copy()
        # now before we return the best combo, we need to remove the evidence
        for key in evidence.keys():
            best_combo.pop(key)
        return (best_combo, best_util)

        # We need to submit a query like P(Y=y| Dec_var=dec_val, evidence

    def vpi(self, potential_evidence, observed_evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.

        :param string potential_evidence: string representing the variable name
        of the variable under evaluation
        :param dict observed_evidence: dict mapping network variables 
        to their observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: float value indicating the VPI(potential | observed)
        """
        # Lets get the values of the potential evidence
        val_list = self.vals[potential_evidence]
        # get meu for observed_evidence
        meu_no_evidence = self.meu(observed_evidence)[1]
        # So we need to create evidence for all possible values and get the meu of that, multipled by the weight of the blank query.
        weights_dict = self.symbolMatch(
            self.bn.predict_proba(observed_evidence), self.cols)
        weight_tuple_list = weights_dict[potential_evidence]
        weights = {}
        for item in weight_tuple_list:

            weights[item[0]] = item[1]
        # okay, so now I can weight I think...
        # so now need to create list of evidence with the potential evidence to give to meu
        list_pot_evidence = []
        for val in val_list:
            temp_dict = observed_evidence.copy()
            temp_dict[potential_evidence] = val
            list_pot_evidence.append(temp_dict.copy())
        # we now have a the list of potential evidence, so now we will get the weight meu sum (note weights are called from their dictionary by the value of the potential evidence)
        meu_weight_sum = 0.0
        for index in range(0, len(list_pot_evidence)):
            meu_weight_sum += (weights[list_pot_evidence[index][potential_evidence]] * (
                (self.meu(list_pot_evidence[index])[1])))
        if meu_weight_sum-meu_no_evidence < 0:
            return 0
        else:
            return meu_weight_sum-meu_no_evidence


class AdAgentTests(unittest.TestCase):

    def test_meu_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv',
                             ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 0}, decision[0])

        self.assertAlmostEqual(2, decision[1], delta=0.01)

    def test_meu_lecture_example_with_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv',
                             ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {"X": 0}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 1}, decision[0])

        self.assertAlmostEqual(2, decision[1], delta=0.01)

        evidence2 = {"X": 1}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"D": 0}, decision2[0])

        self.assertAlmostEqual(2.4, decision2[1], delta=0.01)

    def test_vpi_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv',
                             ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        vpi = ad_engine.vpi("X", evidence)
        self.assertAlmostEqual(0.24, vpi, delta=0.1)

    def test_meu_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv',
                             ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 0}, decision[0])

        self.assertAlmostEqual(746.72, decision[1], delta=0.01)

    def test_meu_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv',
                             ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 1}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 1}, decision[0])

        self.assertAlmostEqual(720.73, decision[1], delta=0.01)

        evidence2 = {"T": 0, "G": 0}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"Ad1": 0, "Ad2": 0}, decision2[0])

        self.assertAlmostEqual(796.82, decision2[1], delta=0.01)

    def test_vpi_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv',
                             ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(20.77, vpi, delta=0.1)

        vpi2 = ad_engine.vpi("F", evidence)
        self.assertAlmostEqual(0, vpi2, delta=0.1)

    def test_vpi_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv',
                             ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 0}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(25.49, vpi, delta=0.1)

        evidence2 = {"G": 1}
        vpi2 = ad_engine.vpi("P", evidence2)
        self.assertAlmostEqual(0, vpi2, delta=0.1)

        evidence3 = {"H": 0, "T": 1, "P": 0}
        vpi3 = ad_engine.vpi("G", evidence3)
        self.assertAlmostEqual(66.76, vpi3, delta=0.1)

        # Facebook yes, Google no


if __name__ == '__main__':
    unittest.main()
