import docplex.cp.model as cp
import numpy as np
import random

def get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_score,
                 product_operation_stage, operations_per_product, full_routing,
                 degree_of_unrelatedness, domain_processing, domain_initial_setup,
                 domain_setup, setup_ratio, bottleneck_dict, time_limit, exec_file, epsilon = 0.00):
    
    d = [i for i in range(res_range[0], res_range[1]+1)]
    product_stages = {p : {s : 1 if s in product_operation_stage[p].values() else 0
                           for s in range(1, nr_stages+1)}  for p in products}
    product_nb_stages = {p : sum(product_stages[p].values()) for p in products}
    possible_machines = ['RES_{}_{}'.format(str(s).zfill(len(str(nr_stages))),str(m).zfill(len(str(res_range[1]))))
                          for s in range(1,nr_stages+1) for m in range(1,res_range[1]+1)]
    possible_machines_stage = {s : ['RES_{}_{}'.format(str(s).zfill(len(str(nr_stages))),str(m).zfill(len(str(res_range[1]))))
                                    for m in range(1,res_range[1]+1)] 
                               for s in range(1,nr_stages+1)}
    product_machine_operation = {p : {m : [o for o in operations_per_product[p] 
                                           if m in possible_machines_stage[product_operation_stage[p][o]]]
                                           for m in possible_machines} for p in products}

    domain_processing_base_list = [i for i in range(domain_processing[0], domain_processing[1]+1)] # TODO adapt?!
    domain_processing_list = [0] + [i for i in range(domain_processing[0], domain_processing[1]+1)]
    domain_initial_setup_list = [i for i in range(domain_initial_setup[0], domain_initial_setup[1]+1)]
    domain_setup_list = [0] + [i for i in range(domain_setup[0], domain_setup[1]+1)]

    stage_machines = {s : 0 for s in range(1,nr_stages+1)}
    s = random.choice(range(1, nr_stages +1))
    tmp_machine_max = min(int(flexibility_score +1), res_range[1])
    tmp_machine_min = max(int(flexibility_score -1), res_range[0])
    stage_machines[s] = random.randint(tmp_machine_min, tmp_machine_max)

    if bottleneck_dict is None:
        bottleneck_dict = {s : 1 for s in range(1, nr_stages+1)}

    # randomize some machine assigments         
    if not full_routing:
        product_machine_assigments = {}
        # TODO randomize
        # random.randint()

    # TODO randomize some entries
    init_setup = {m : {p : -1 for p in products} for m in possible_machines}
    # init_setup = {m : {p : random.randint(domain_initial_setup_list[0], domain_initial_setup_list[-1]) for p in products} for m in possible_machines}

    mdl = cp.CpoModel()
    # variables
    
    M_s =  {s : cp.integer_var(name='M_{}'.format(s), domain = d)
            for s in range(1,nr_stages+1) 
            }
    Y_m = {m : cp.binary_var(name='Y_{}'.format(m))
            for m in possible_machines
            }

    X_mi = {(m, i) : cp.binary_var(name='X_{}{}'.format(m, i)) 
            for m in possible_machines 
            for i in products
            }
    
    P_Base_io = {(i,o) : cp.integer_var(name='P_{}{}'.format(i,o), domain = domain_processing_base_list)
                  for i in products
                  for o in operations_per_product[i]
                }
    
    P_iom =  {(i,o,m) : cp.integer_var(name='P_{}{}{}'.format(i,o,m), domain = domain_processing_list)
               for i in products
               for o in operations_per_product[i]
               for m in possible_machines_stage[product_operation_stage[i][o]]
                }
    
    S_initial_mi =  {(m,i) : cp.integer_var(name='S_{}{}'.format(m,i), domain = domain_initial_setup_list)
                      for i in products
                      for m in possible_machines
                    }
    
    S_mij =  {(m,i,j) : cp.integer_var(name='S_{}{}{}'.format(m,i,j), domain = domain_setup_list)
               for i in products
               for j in products
               for m in possible_machines
               }
    
    # W_m = {m : cp.integer_var(name='W_{}'.format(m)) for m in possible_machines}

    # constraints

    if full_routing:
        mdl.add(cp.equal(product_stages[i][s] * Y_m[m], X_mi[m,i])
                          for i in products for s in range(1, nr_stages+1) for m in possible_machines_stage[s])
    else:
        mdl.add(cp.greater(cp.sum(product_stages[i][s] * cp.sum(Y_m[m] for m in possible_machines_stage[s]) for s in range(1, nr_stages+1)),
                            cp.sum(X_mi[m,i] for m in possible_machines))
                            for i in products)
        
    # only machines active that have at least one product assigned
    mdl.add(cp.less_or_equal(Y_m[m], cp.sum(X_mi[m,i] for i in products)) for m in possible_machines)

    # only assigment to machine if present
    mdl.add(cp.greater_or_equal(Y_m[m], X_mi[m,i]) for m in possible_machines for i in products)

    # enforce random machine drawn
    mdl.add(cp.equal(M_s[s], stage_machines[s])  for s in range(1, nr_stages+1) if stage_machines[s] > 0) 
    
    # existing machines on each stage has to equal to number of machines on stage
    mdl.add(cp.equal(cp.sum(Y_m[m] for m in possible_machines_stage[s]), M_s[s]) for s in range(1, nr_stages+1))

    # symmetry breaker
    mdl.add(cp.greater_or_equal(Y_m[m], Y_m[possible_machines_stage[s][i+1]]) for s in range(1, nr_stages+1) for i,m in enumerate(possible_machines_stage[s][:-1]))

    # range of assigments
    # >=1 if present at stage
    mdl.add(cp.greater_or_equal(cp.sum(X_mi[m, i] for m in possible_machines_stage[s]), product_stages[i][s]) for s in range(1, nr_stages+1) for i in products)
    # # <= M_s * present at stage | done in contraint above
    # mdl.add(cp.less_or_equal(cp.sum(X_mi[m, i] for m in possible_machines_stage[s]), M_s[s] * product_stages[i][s]) for s in range(1, nr_stages+1) for i in products)
    
    # hard constraint,
    mdl.add(cp.equal(
        cp.sum(X_mi[m,i] for m in possible_machines), int(flexibility_score * product_nb_stages[i]))
        for i in products)

    # TODO add epsilon for constraint
    # mdl.add(cp.greater_or_equal(
    #     cp.sum(X_mi[m,i] for m in possible_machines),
    #     (1 - epsilon) * flexibility_score * product_nb_stages[i])
    #     for i in products)
    
    # mdl.add(cp.less_or_equal(
    #     cp.sum(X_mi[m,i] for m in possible_machines),
    #     (1 + epsilon) * flexibility_score * product_nb_stages[i])
    #     for i in products)

    
    # relate processing time to base for same operations
    mdl.add(cp.less_or_equal(
        P_Base_io[i,o] * (1 - degree_of_unrelatedness) * X_mi[m,i] , P_iom[i,o,m])
        for i in products for o in operations_per_product[i]
        for m in possible_machines_stage[product_operation_stage[i][o]]
        )
    mdl.add(cp.greater_or_equal(
        P_Base_io[i,o] * (1 + degree_of_unrelatedness) * X_mi[m,i], P_iom[i,o,m])
        for i in products for o in operations_per_product[i]
        for m in possible_machines_stage[product_operation_stage[i][o]]
        )
    
    # non present products get zero processing time
    mdl.add(cp.less_or_equal(P_iom[i,o,m], domain_processing[1] * X_mi[m,i]) for i in products for o in operations_per_product[i]
            for m in possible_machines_stage[product_operation_stage[i][o]]
            )
    
    # setups
    # enforcing zero for same product
    mdl.add(cp.equal(S_mij[m,i,j], 0) for m in possible_machines for i in products for j in products if i == j)

    # enforcing zero for not present products
    mdl.add(cp.less_or_equal(S_mij[m,i,j], domain_setup[1] * X_mi[m,i] * X_mi[m,j] ) for m in possible_machines for i in products for j in products if i != j)

    # asymmetry using big M with two different values
    mdl.add(cp.diff(S_mij[m,i,j] +  (domain_setup[1] + 1) * (1 - X_mi[m,i]), S_mij[m,j,i] + (domain_setup[1] + 2) * (1 - X_mi[m,j]))
            for m in possible_machines for i in products for j in products if i != j)

    # TODO is this needed? could have zero setup between products!?
    mdl.add(cp.diff(S_mij[m,i,j], (X_mi[m,i] + X_mi[m,j] - 2)) for m in possible_machines for i in products for j in products if i != j)

    # triangle inequality
    mdl.add(cp.less_or_equal(S_mij[m,i,j], S_mij[m,i,k] + S_mij[m,k,j] + domain_setup[1] * (1 - X_mi[m,k])) for m in possible_machines 
            for i in products for j in products for k in products
            if i != j and i != k and j != k)

    # initial setup
    mdl.add(cp.equal(S_initial_mi[m,i], init_setup[m][i] * X_mi[m,i]) for m in possible_machines for i in products if init_setup[m][i] >= 0)

    # setup ratio == average setuptime / avg processing time
    mdl.add(cp.equal(
        cp.sum(P_iom[i,o,m] for i in products for o in operations_per_product[i] for m in possible_machines_stage[product_operation_stage[i][o]]) * setup_ratio /
          cp.sum(X_mi[m,i] for m in possible_machines for i in products),
        (cp.sum(S_mij[m,i,j] for m in possible_machines for i in products for j in products) + 
        cp.sum(S_initial_mi[m,i] for m in possible_machines for i in products)) /
        (cp.sum(cp.sum(X_mi[m,i] for i in products)**2 + cp.sum(X_mi[m,i] for i in products) for m in possible_machines))
        ))
    
    # workload of each machine
    # mdl.add(cp.equal(W_m[m], cp.sum(prod_arrival_prob[i] * (P_iom[i,o,m] +
    #                                                         (
    #                         (cp.sum(S_mij[m,i,j] for j in products) + S_initial_mi[m,i]) /
    #                         (cp.sum(X_mi[r,i] for r in possible_machines_stage[product_operation_stage[i][o]]) + 1 - Y_m[m]) 
    #                     )
    #                     ) /
    #                     (1 - Y_m[m] + cp.sum(X_mi[r,i] for r in possible_machines_stage[product_operation_stage[i][o]]))
    #                     for i in products for o in product_machine_operation[i][m])) 
    #                 for m in possible_machines)

    Workload = {s : bottleneck_dict[s] * cp.sum(
                        cp.sum(prod_arrival_prob[i] * (P_iom[i,o,m] +
                                                            (
                            (cp.sum(S_mij[m,j,i] for j in products) + S_initial_mi[m,i]) /
                            (cp.sum(X_mi[m,j] for j in products) + 1) 
                        )
                        ) /
                        (1 - Y_m[m] + cp.sum(X_mi[r,i] for r in possible_machines_stage[product_operation_stage[i][o]]))
                        for i in products for o in product_machine_operation[i][m]) 
                    for m in possible_machines_stage[s])
                    / M_s[s]
                for s in range(1, nr_stages+1)
                }

    max_workload = cp.max(Workload[s] for s in range(1, nr_stages+1))
    min_workload = cp.min(Workload[s] for s in range(1, nr_stages+1))
    # min_workload = cp.min(Workload[s] + (max_workload * (1 - Y_m[m])) for m in possible_machines)

    mdl.add(cp.less_or_equal(max_workload, (1 + epsilon) * min_workload))
    

        # mdl.add(cp.minimize(violation))

    res = mdl.solve(TimeLimit= time_limit, 
                    agent = "local",
                    execfile= exec_file,
                    LogVerbosity = "Quiet",
                    trace_log = False
                    )
    if res:
        nb_machines = np.zeros(nr_stages)
        for n in M_s:
            nb_machines[n-1] = res[M_s[n]]

        routing = {}
        if not full_routing:
            for x in X_mi:
                if res[X_mi[x]] == 1:
                    for o in product_machine_operation[x[1]][x[0]]:
                        routing.setdefault(x[1], {}).setdefault(o, []).append(str(x[0]))

        processing_times = {}
        setup_times = {}
        initial_setup_times = {}
        for p in P_iom:
            if res[X_mi[p[2], p[0]]] == 1:
                processing_times.setdefault(p[0], {}).setdefault(p[1], {}).setdefault(p[2], res[P_iom[p]])
        for s in S_mij:
            if res[X_mi[s[0], s[1]]] == 1 and res[X_mi[s[0], s[2]]] == 1: #TODO might have to remove to not break sim
                setup_times.setdefault(s[0], {}).setdefault(s[1], {}).setdefault(s[2], res[S_mij[s]])
        for i in S_initial_mi:
            if res[X_mi[i[0], i[1]]] == 1:
                initial_setup_times.setdefault(i[0], {}).setdefault(i[1], res[S_initial_mi[i]])

        workload_machine = {m: cp.sum(prod_arrival_prob[i] * (res[P_iom[i,o,m]] +
                                                            (
                            (cp.sum(res[S_mij[m,i,j]] for j in products) + res[S_initial_mi[m,i]]) /
                            (cp.sum(res[X_mi[m,j]] for j in products) + 1) 
                            )
                            ) /
                            (1 - res[Y_m[m]] + cp.sum(res[X_mi[r,i]] for r in possible_machines_stage[product_operation_stage[i][o]]))
                            for i in products for o in product_machine_operation[i][m]) 
                        for m in possible_machines if res[Y_m[m]] == 1
                        }
        max_workload = max(workload_machine.values())
        for key in workload_machine.keys():
            workload_machine[key] /= max_workload
        return nb_machines, routing, processing_times, setup_times, initial_setup_times, \
                workload_machine, max_workload, epsilon
    else:
        # if epsilon < 0.05:
        #     epsilon += 0.01
        #     get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_score,
        #                 product_operation_stage, operations_per_product, full_routing,
        #                 degree_of_unrelatedness, domain_processing, domain_initial_setup,
        #                 domain_setup, setup_ratio, bottleneck_dict = bottleneck_dict, epsilon = epsilon)
        # else:
        failed_constraints = mdl.refine_conflict()
        print(failed_constraints)
        raise ValueError("No solution found, unfavorable parameters?!")

def create_layout(nr_stages:int, nr_products:int, prod_arrival_prob:list, res_range:tuple, skipping_prob:float,
                    full_routing:bool, flexibility_target:float, 
                    processing_range:tuple, initial_setup_range:tuple,
                    setup_range:tuple, ratio_setup_processing:float,
                    degree_of_unrelatedness:float, seed, bottleneck_dict, time_limit, exec_file
                    ):
    """
    Creates an instance of the flow shop problem with the given parameters.
    """


    assert len(prod_arrival_prob) == nr_products, "Arrival probabilities do not match number of products"
    assert all([p != 0 for p in prod_arrival_prob]), "Arrival probabilities of zero is not supported"
    assert sum(prod_arrival_prob) == 1, "Arrival probabilities do not sum to 1"
    assert nr_stages > 0, "Invalid number of stages"
    assert nr_products > 0, "Invalid number of products"
    assert 0 <= res_range[0] <= res_range[1], "Invalid range for number of machines"
    assert 0 <= skipping_prob <= 1, "Invalid skipping probability"
    assert 0 <= flexibility_target, "Invalid flexibility target"
    assert 0 <= degree_of_unrelatedness <= 1, "Invalid degree of unrelatedness"
    assert 0 <= processing_range[0] <= processing_range[1], "Invalid range for processing times"
    assert 0 <= initial_setup_range[0] <= initial_setup_range[1], "Invalid range for initial setup times"
    assert 0 <= setup_range[0] <= setup_range[1], "Invalid range for setup times"
    assert 0 <= ratio_setup_processing, "Invalid ratio of setup to processing times"


    prod_arrival_prob = {'PRODUCT_{}'.format(str(i+1).zfill(len(str(nr_products)))) : p for i, p in enumerate(prod_arrival_prob)}
    # generate Layout
    # Resources
    resources = []
    stage_resource = {}
    resource_stage = {}

    if seed:
        random.seed(seed)

    products = ["PRODUCT_" + str(i+1).zfill(len(str(nr_products))) for i in range(nr_products)]
    operations = set()
    product_operation_stage = {}
    operations_per_product = {k :[] for k in products}
    operation_machine_compatibility = {}
    for p in products:
        stages_present = 0
        for i in range(1, nr_stages+1):
            if random.random() >= skipping_prob or (stages_present == 0 and i == nr_stages):
                stages_present += 1
                op_str = 'OP_{}'.format(str(i).zfill(len(str(nr_stages))))
                operations.add(op_str)
                product_operation_stage.setdefault(p, {}).setdefault(op_str, i)
                operations_per_product[p].append(op_str)
    operations = list(operations)

    number_machine_stages, routing, processing_times, setup_times \
        ,initial_setups, workload_machine, max_workload, epsilon = \
            get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_target,
                         product_operation_stage, operations_per_product, full_routing,
                         degree_of_unrelatedness, processing_range, initial_setup_range,
                         setup_range, ratio_setup_processing, bottleneck_dict, time_limit, exec_file)
    
    # get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_score, product_operation_stage, operations_per_product, full_routing,
    # degree_of_unrelatedness, domain_processing, domain_initial_setup, domain_setup, setup_ratio):
    
    for i in range(1, nr_stages+1):
        tmp_res = []
        for j in range(int(number_machine_stages[i-1])):
            tmp_name = 'RES_{}_{}'.format(str(i).zfill(len(str(nr_stages))),str(j+1).zfill(len(str(res_range[1]))))
            resources.append(tmp_name)
            tmp_res.append(tmp_name)
            resource_stage[tmp_name] = i
        stage_resource[i] = tmp_res

    if full_routing:
        for p in products:
            for op_str in operations_per_product[p]:
                operation_machine_compatibility.setdefault(p, {}).setdefault(op_str, stage_resource[product_operation_stage[p][op_str]])
    else:
        operation_machine_compatibility = routing

    # generate dict
    file_dict = {}
    file_dict['resources'] = resources
    file_dict['resource_stage'] = resource_stage
    file_dict['stage_resource'] = stage_resource
    file_dict['NrStages'] = nr_stages
    file_dict['products'] = products
    file_dict['operations'] = operations
    file_dict['product_operation_stage'] = product_operation_stage
    file_dict['operations_per_product'] = operations_per_product
    file_dict['operation_machine_compatibility'] = operation_machine_compatibility
    file_dict['processing_time'] = processing_times
    file_dict['setup_time'] = setup_times
    file_dict['initial_setup'] = initial_setups
    file_dict['arrival_prob'] = prod_arrival_prob

    file_dict['workload_machine'] = workload_machine
    file_dict['max_workload'] = max_workload
    file_dict['flexibility_score'] = flexibility_target
    file_dict['epsilon'] = epsilon
    file_dict['seed_layout'] = seed


    return file_dict


if __name__ == "__main__":
    create_layout(nr_stages= 4, nr_products = 4, prod_arrival_prob = [0.3, 0.3, 0.2, 0.2], res_range = (2,5), skipping_prob = 0.2,
                    full_routing = False, flexibility_target = 2.0, seed = 42
                    )