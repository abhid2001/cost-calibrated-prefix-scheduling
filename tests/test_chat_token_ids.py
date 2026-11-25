import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import chat_token_ids
class TestChatTokenIds(unittest.TestCase):
    def test_list_and_mapping(self):
        class Tokenizer:
            def __init__(self, result): self.result=result
            def apply_chat_template(self,*args,**kwargs): return self.result
        for value in ([1,2], {'input_ids':[1,2]}, {'input_ids':[[1,2]]}):
            self.assertEqual(chat_token_ids(Tokenizer(value),[]),[1,2])
        with self.assertRaises(ValueError): chat_token_ids(Tokenizer({'input_ids':'invalid'}),[])
