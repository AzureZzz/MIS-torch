

def show_net_structure(net):
    for name, module in net._modules.items():
        print(name, ":", module)