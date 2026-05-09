from embedding import Embedding

emb = Embedding("BAAI/bge-m3")

text = "123\n2 2\n\n2　2,2.2，2。2"
tokens = emb.encode(text)
print(tokens)
print(emb.decode(tokens))