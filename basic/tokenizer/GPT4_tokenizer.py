

class BasicTokenizer():
  def __init__(self):
    super().__init__()
    # maps merged pair (p0, p1) -> new token id; built during training
    self.merges = {}

  # count occurrences of each consecutive pair in the token sequence
  def get_stats(self, ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
      counts[pair] = counts.get(pair, 0) + 1
    return counts

  # replace every occurrence of pair in ids with new_token_id
  def merge(self, ids, pair, new_token_id):
    new_ids = []
    i = 0
    while i < len(ids):
      if i != len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
        new_ids.append(new_token_id)
        i += 2
      else:
        new_ids.append(ids[i])
        i += 1
    return new_ids

  def train(self, text, vocab_size, verbose=False):
    # token ids 0-255 are reserved for raw bytes; new tokens start at 256
    ids = list(text.encode("utf-8"))
    merge_count = vocab_size - 256

    for i in range(merge_count):
      # find the most frequent pair and assign it a new token id
      stats = self.get_stats(ids)
      pair = max(stats, key=stats.get)
      new_token_id = i + 256
      ids = self.merge(ids, pair, new_token_id)
      self.merges[pair] = new_token_id

  def encode(self, text):
    # start with raw utf-8 bytes, then apply merges in training order
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
      stats = self.get_stats(ids)
      # apply the merge that was learned earliest (lowest token id)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
      if pair not in self.merges:
        break
      ids = self.merge(ids, pair, self.merges[pair])
    return ids

  def decode(self, ids):
    # build vocab: map each token id back to its byte sequence
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]

    tokens = b"".join(vocab[i] for i in ids)
    return tokens.decode("utf-8", errors="replace")
  
  
def main():
  with open("taylorswift.txt", "r") as f:
    text = f.read()
  
  # print(len(text))
  # tokens = list(text.encode("utf-8"))
  # print(tokens)
  basic_tokenizer = BasicTokenizer()
  basic_tokenizer.train(text, 276)
  print(text == basic_tokenizer.decode(basic_tokenizer.encode(text)))

if __name__ == "__main__":
    main()