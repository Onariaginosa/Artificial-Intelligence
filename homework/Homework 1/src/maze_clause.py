'''
maze_clause.py

Specifies a Propositional Logic Clause formatted specifically
for Grid Maze Pathfinding problems. Clauses are a disjunction of
MazePropositions (2-tuples of (symbol, location)) mapped to
their negated status in the sentence.
'''
import unittest

class MazeClause:
    
    def __init__(self, props):
        """
        Constructor parameterized by the propositions within this clause;
        argument props is a list of MazePropositions, like:
        [(("X", (1, 1)), True), (("X", (2, 1)), True), (("Y", (1, 2)), False)]
        
        :props: a list of tuples formatted as: (MazeProposition, NegatedBoolean)
        """
        self.valid = False
        props = list(set(props))
        for prop1 in props:
            for prop2 in props:
                if prop1[0] == prop2[0] and prop1[1] != prop2[1]:
                    self.valid = True
                    break
            if self.valid:
                break
        if (self.valid):
            self.props = dict([])
        else:
            self.props = dict(props)
    
    def get_prop(self, prop):
        """
        Returns:
          - None if the requested prop is not in the clause
          - True if the requested prop is positive in the clause
          - False if the requested prop is negated in the clause
          
        :prop: A MazeProposition as a 2-tuple formatted as: (Symbol, Location),
        for example, ("P", (1, 1))
        """
        for p in self.props:
            if(p == prop):
                return True if self.props.get(p) else False
        return None
    
    def is_valid(self):
        """
        Returns:
          - True if this clause is logically equivalent with True
          - False otherwise
        """
        return self.valid
    
    def is_empty(self):
        """
        Returns:
          - True if this is the Empty Clause
          - False otherwise
        (NB: valid clauses are not empty)
        """
        return len(self.props) == 0 and not self.valid
    
    def __eq__(self, other):
        """
        Defines equality comparator between MazeClauses: only if they
        have the same props (in any order) or are both valid
        """
        return self.props == other.props and self.valid == other.valid
    
    def __hash__(self):
        """
        Provides a hash for a MazeClause to enable set membership
        """
        # Hashes an immutable set of the stored props for ease of
        # lookup in a set
        return hash(frozenset(self.props.items()))
    
    # Hint: Specify a __str__ method for ease of debugging (this
    # will allow you to "print" a MazeClause directly to inspect
    # its composite literals)
    def __str__ (self):
        clause = f"Valid: {self.valid} | Props: [  "
        for prop in self.props:
            if not self.props.get(prop):
                clause += "!"
            clause += f"{prop} V "
        clause = clause[:-2]
        return clause + " ]"
    
    @staticmethod
    def resolve(c1, c2):
        """
        Returns a set of MazeClauses that are the result of resolving
        two input clauses c1, c2 (Hint: result will only ever be a set
        of 0 or 1 MazeClause, but it being a set is convenient for the
        inference engine) (Hint2: returning an empty set of clauses
        is different than returning a set containing the empty clause /
        contradiction)
        
        :c1: A MazeClause to resolve with c2
        :c2: A MazeClause to resolve with c1
        """
        clauses = set()
        for p1 in set(c1.props):
            for p2 in c2.props:
                if p1 == p2 and not c1.props.get(p1) == c2.props.get(p2):
                    clause1 = [(k1, v1) for k1, v1 in c1.props.items() if k1 != p1] 
                    clause2 = [(k2, v2) for k2, v2 in c2.props.items() if k2 != p2]
                    clause = [*clause1, *clause2]
                    clause = MazeClause(clause)
                    if clause != c1 and clause != c2 and not clause.valid:
                        clauses.add(clause)
        return clauses



class MazeClauseTests(unittest.TestCase):

    def test_mazeprops1(self):
        mc = MazeClause([(("X", (1, 1)), True), (("X", (2, 1)), True), (("Y", (1, 2)), False)])
        self.assertTrue(mc.get_prop(("X", (1, 1))))
        self.assertTrue(mc.get_prop(("X", (2, 1))))
        self.assertFalse(mc.get_prop(("Y", (1, 2))))
        self.assertTrue(mc.get_prop(("X", (2, 2))) is None)
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops2(self):
        mc = MazeClause([(("X", (1, 1)), True), (("X", (1, 1)), True)])
        self.assertTrue(mc.get_prop(("X", (1, 1))))
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops3(self):
        mc = MazeClause([(("X", (1, 1)), True), (("Y", (2, 1)), True), (("X", (1, 1)), False)])
        self.assertTrue(mc.is_valid())
        self.assertTrue(mc.get_prop(("X", (1, 1))) is None)
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops4(self):
        mc = MazeClause([])
        self.assertFalse(mc.is_valid())
        self.assertTrue(mc.is_empty())
        
    def test_mazeprops5(self):
        # No resolution
        mc1 = MazeClause([(("X", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops6(self):
        mc1 = MazeClause([(("X", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([]) in res)
        
    def test_mazeprops7(self):
        # resolves properly
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (2, 2)), True)])
        res = MazeClause.resolve(mc1, mc2)
        # res should equal 
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([(("Y", (1, 1)), True), (("Y", (2, 2)), True)]) in res)
        
    def test_mazeprops8(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)])
        res = MazeClause.resolve(mc1, mc2)
        print ('Res:')
        for c in res:
            print(c)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops9(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False), (("Z", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True), (("W", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops10(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False), (("Z", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), False), (("W", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True), (("W", (1, 1)), False)]) in res)
        
if __name__ == "__main__":
    unittest.main()
    