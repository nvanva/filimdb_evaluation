import torch
from fire import Fire

from translit import PositionalEncoding, LrScheduler, MultiHeadAttention,\
    compositional_mask

def test_positional_encoding():
    pe = PositionalEncoding(max_len=3, hidden_size=4)
    res_1 = torch.tensor([[[ 0.0000,  1.0000,  0.0000,  1.0000],
                           [ 0.8415,  0.5403,  0.0100,  0.9999],
                           [ 0.9093, -0.4161,  0.0200,  0.9998]]])
    # print(pe.pe - res_1)
    assert torch.all(torch.abs(pe.pe - res_1) < 1e-4).item()
    print('Test is passed!')

def test_lr_scheduler():
    lrs_type = 'warmup,decay_linear'
    warmup_steps_part =  0.1
    lr_peak = 3e-4
    sch = LrScheduler(100, type=lrs_type, warmup_steps_part=warmup_steps_part,
                      lr_peak=lr_peak)
    assert sch.learning_rate(step=5) - 15e-5 < 1e-6
    assert sch.learning_rate(step=10) - 3e-4 < 1e-6
    assert sch.learning_rate(step=50) - 166e-6 < 1e-6
    assert sch.learning_rate(step=100) - 0. < 1e-6
    print('Test is passed!')

def test_multi_head_attention():
    mha = MultiHeadAttention(n_heads=1, hidden_size=5, dropout=None)
    # batch_size == 2, sequence length == 3, hidden_size == 5
    # query = torch.arange(150).reshape(2, 3, 5)
    query = torch.tensor([[[[ 0.64144618, -0.95817388,  0.37432297,  0.58427106,
          -0.94668716]],
        [[-0.23199289,  0.66329209, -0.46507035, -0.54272512,
          -0.98640698]],
        [[ 0.07546638, -0.09277002,  0.20107185, -0.97407381,
          -0.27713414]]],
       [[[ 0.14727783,  0.4747886 ,  0.44992016, -0.2841419 ,
          -0.81820319]],
        [[-0.72324994,  0.80643179, -0.47655449,  0.45627872,
           0.60942404]],
        [[ 0.61712569, -0.62947282, -0.95215713, -0.38721959,
          -0.73289725]]]])
    key = torch.tensor([[[[-0.81759856, -0.60049991, -0.05923424,  0.51898901,
          -0.3366209 ]],
        [[ 0.83957818, -0.96361722,  0.62285191,  0.93452467,
           0.51219613]],
        [[-0.72758847,  0.41256154,  0.00490795,  0.59892503,
          -0.07202049]]],
       [[[ 0.72315339, -0.49896314,  0.94254637, -0.54356006,
          -0.04837949]],
        [[ 0.51759322, -0.43927061, -0.59924184,  0.92241702,
          -0.86811696]],
        [[-0.54322046, -0.92323003, -0.827746  ,  0.90842783,
           0.88428119]]]])
    value = torch.tensor([[[[-0.83895431,  0.805027  ,  0.22298283, -0.84849915,
          -0.34906026]],
        [[-0.02899652, -0.17456128, -0.17535998, -0.73160314,
          -0.13468061]],
        [[ 0.75234265,  0.02675947,  0.84766286, -0.5475651 ,
          -0.83319316]]],
       [[[-0.47834413,  0.34464645, -0.41921457,  0.33867964,
           0.43470836]],
        [[-0.99000979,  0.10220893, -0.4932273 ,  0.95938905,
           0.01927012]],
        [[ 0.91607137,  0.57395644, -0.90914179,  0.97212912,
           0.33078759]]]])
    query = query.float().transpose(1,2)
    key = key.float().transpose(1,2)
    value = value.float().transpose(1,2)

    x,_ = torch.max(query[:,0,:,:], axis=-1)
    mask = compositional_mask(x)
    mask.unsqueeze_(1)
    for n,t in [('query', query), ('key', key), ('value', value), ('mask', mask)]:
        print(f'Name: {n}, shape: {t.size()}')
    with torch.no_grad():
        output, attn_weights = mha.attention(query, key, value, mask=mask)
    assert output.size() == torch.Size([2,1,3,5])
    assert attn_weights.size() == torch.Size([2,1,3,3])

    truth_output = torch.tensor([[[[-0.8390,  0.8050,  0.2230, -0.8485, -0.3491],
          [-0.6043,  0.5212,  0.1076, -0.8146, -0.2870],
          [-0.0665,  0.2461,  0.3038, -0.7137, -0.4410]]],
        [[[-0.4783,  0.3446, -0.4192,  0.3387,  0.4347],
          [-0.7959,  0.1942, -0.4652,  0.7239,  0.1769],
          [-0.3678,  0.2868, -0.5799,  0.7987,  0.2086]]]])
    truth_attn_weights = torch.tensor([[[[1.0000, 0.0000, 0.0000],
          [0.7103, 0.2897, 0.0000],
          [0.3621, 0.3105, 0.3274]]],
        [[[1.0000, 0.0000, 0.0000],
          [0.3793, 0.6207, 0.0000],
          [0.2642, 0.4803, 0.2555]]]])
    # print(torch.abs(output - truth_output))
    # print(torch.abs(attn_weights - truth_attn_weights))
    assert torch.all(torch.abs(output - truth_output) < 1e-4).item()
    assert torch.all(torch.abs(attn_weights - truth_attn_weights) < 1e-4).item()
    print('Test is passed!')

if __name__=='__main__':
    Fire()
