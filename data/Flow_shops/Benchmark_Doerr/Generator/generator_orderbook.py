import random

def create_orderbook(problem_instance, number_orders, tightness_due_dates = 1, seed = None):
    if seed is not None: random.seed = seed
    probabilities = problem_instance['arrival_prob']
    prob_vec = [p for p in probabilities.values()]
    prod_vec = [p for p in probabilities.keys()]

    products_drawn = random.choices(prod_vec, weights = prob_vec, k = number_orders)

    min_due = {p : calc_min_due_date(problem_instance, p) for p in prod_vec}
    total_time = problem_instance['max_workload'] * number_orders

    orders = {}
    for i,j in enumerate(products_drawn):
        name = 'ORD_' + str(i+1).zfill(len(str(number_orders)))
        due_date = (random.random() * total_time) + (min_due[j] * tightness_due_dates)
        orders[name] = {'due_date' : due_date, 'product' : j}

    problem_instance["orders"] = orders
    return problem_instance

def calc_min_due_date(instance, product):
    time = 0
    for o in instance['operations_per_product'][product]:
        min_time_op = min(
            instance['processing_time'][product][o][m]['mean'] + 
            instance['initial_setup'][m][product]['mean']
            if instance['initial_setup'][m][product]['type'] != 'Nonexisting'
            else instance['processing_time'][product][o][m]['mean']
            for m in instance['operation_machine_compatibility'][product][o]
        )
        time += min_time_op
    return time