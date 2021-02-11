'''
maze_knowledge_base.py

Specifies a simple, Conjunctive Normal Form Propositional
Logic Knowledge Base for use in Grid Maze pathfinding problems
with side-information.
'''
from maze_clause import MazeClause
import unittest
import copy

class MazeKnowledgeBase:
    
    def __init__ (self):
        self.clauses = set()

    def __str__(self):
        string = 'KnowledgeBase: {  \n'
        for clause in self.clauses:
            string += MazeClause.__str__(clause) + ', \n'
        return string + '}'
    
    def tell (self, clause):
        """
        Adds the given clause to the CNF MazeKnowledgeBase
        Note: we expect that no clause added this way will ever
        make the KB inconsistent (you need not check for this)
        """
        self.clauses.add(clause)
        
    def ask (self, query):
        """
        Given a MazeClause query, returns True if the KB entails
        the query, False otherwise
        """
        temp_kb = copy.deepcopy(self)
        for prop in query.props:
            # get the clauses
            clause = [(prop, False if query.props.get(prop) else True)]
            clause = MazeClause(clause)
            temp_kb.tell(clause)
        while True:
            kb = list(temp_kb.clauses)
            for i in range(0, len(kb)):
                for j in range(i+1, len(kb)):
                    res = MazeClause.resolve(kb[i], kb[j])
                    if MazeClause([]) in res:
                        return True
                    for clause in res:
                            temp_kb.tell(clause)

            if len(temp_kb.clauses) == len(kb):
                return False


class MazeKnowledgeBaseTests(unittest.TestCase):
    def test_mazekb1(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("X", (1, 1)), True)])))
        
    def test_mazekb2(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (1, 1)), True)])))

        
    def test_mazekb3(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)]))
        kb.tell(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("W", (1, 1)), True)])))
        self.assertFalse(kb.ask(MazeClause([(("Y", (1, 1)), False)])))


if __name__ == "__main__":
    unittest.main()