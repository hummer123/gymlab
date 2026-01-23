
def str_key(*args):
    ''' 将多个参数转换为字符串键 '''
    new_arg = []
    for arg in args:
        if type(arg) in [list, tuple]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return '_'.join(new_arg)

