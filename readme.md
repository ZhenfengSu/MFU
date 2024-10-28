# MFU Calculate For Deit Model

## Code Implementation Approach
Refer to [link](https://kadxsrk5f5d.feishu.cn/wiki/YSoZwjxMNifjjdkLZzFc1dnQn0e?from=from_copylink)

## Execution

### Environment Setup
```bash
pip install torch torchvision
```

### Command
```bash
# Given the FLOPs of the model, output the MFU for each model, unit in MACs  
python main.py --flops 50 --output flops_50G.txt
# Given the params of the model, output the MFU for each model, unit in M
python main.py --params 50 --output params_50M.txt
```