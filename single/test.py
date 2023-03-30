import torch
a = torch.tensor(
    [[[1,2,3],
      [2,3,4]],
     [[3,4,5],
      [4,5,6]]]
)

b = torch.tensor(
    [[[1,2,3],
      [2,3,4]]]
)
print(a,b,b+a)