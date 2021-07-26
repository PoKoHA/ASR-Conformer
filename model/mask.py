import torch

"""
Padding Mask 하는 곳
"""
def get_attn_pad_mask(inputs, input_lengths, expand_length):

    def get_transformer_non_pad_mask(inputs, input_lengths):
        batch_size = inputs.size(0)

        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size()) # B x T
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1]) # B x T
        else:
            raise ValueError("Input Shape Error")

        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0

        return non_pad_mask
    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths)
    # print("non_pad_mask: ", non_pad_mask.size())
    pad_mask = non_pad_mask.lt(1)
    # torch.lt(input, other, *, out=None) → Tensor
    # 즉 1보다 작은 값들은 True 1보다 크거나 같으면 False
    # print("pad_mask: ", pad_mask.unsqueeze(1).size())
    # print("expand_length: ", expand_length)
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    # repeat와 expand 차이점: 결과는 같음
    # 하지만 expand 경우 expand 하고 싶은 dim 이 1이여야하고 장점으로는 추가 memory 사용 X
    # repeat 경우 1 이상일 때 사용해야 하고 추가 메모리 사용

    return attn_pad_mask

"""
Decoder 의 auto-regression property 를 위해서 사용할 예정
"""
def get_attn_subsequent_mask(seq, args):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool().cuda(args.gpu)
    # torch.triu(input, diagonal=0, *, out=None) 경우
    # [1 1 1 1 1]
    # [0 1 1 1 1]
    # [0 0 1 1 1]
    # [0 0 0 1 1]
    # [0 0 0 0 1]
    # upper triangular part 를 만듬 diagonal = 1 인 경우 (0,1)에서 시작
    # -1 인 경우 (1,0)에서 시작
    subsequent_mask = subsequent_mask.cuda(args.gpu)

    return subsequent_mask