import unittest
from fp_research.sites import refine


class TestSites(unittest.TestCase):
    def test_same_intent_different_runtime_locations(self):
        ids=["a/test", "b/test", "c/test"]
        sem=dict(groups={"first":ids},pairs=dict.fromkeys(ids))
        sites={k:{"site":["error","product",line]} for k,line in zip(ids,["A:2","A:2","B:7"])}
        self.assertEqual(sorted(refine(sem,sites).values()),[ids[:2],ids[2:]])

    def test_no_stack_does_not_invent_a_location(self):
        sem=dict(groups={"group":["a"]},pairs={"a":None})
        self.assertEqual(refine(sem,{"a":{"site":None}}),sem["groups"])
