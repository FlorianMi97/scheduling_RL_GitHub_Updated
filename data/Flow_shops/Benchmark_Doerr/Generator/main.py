import pathlib

import generator

if __name__ == "__main__":
    exec_file = "C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe"
    dirname = pathlib.Path(__file__).parent.parent.resolve()
    target = str(dirname) + '/Instances/'

    gen = generator.Generator(target, exec_file)
    gen.generate_instance('default_problem', layout_dict={'seed': 42}, orderbook_dict={'tightness_due_dates' : 1, 'seed' : 42})