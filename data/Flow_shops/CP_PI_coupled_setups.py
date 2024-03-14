import docplex.cp.model as cp
import json
import pathlib
import os
import numpy as np
import random

# requires licence for cplex studio and a local installation 

#-----------------------------------------------------------------------------
# TODO Data Reader for new data format/instances!
#-----------------------------------------------------------------------------


def CPsolution(instance, sample, objective, stoppingCriteria = "", saveGantt = False):
    """
    create CP model based on inputs and returns objective value (and solution?)
    """

    data = instance
    dataS = sample

    MACHINES = data["resources"]
    NB_MACHINES = len(MACHINES)

    tmpT = data["products"]
    NB_TYPES = len(tmpT)

    tmpJ = data["orders"].keys()
    NB_JOBS = len(tmpJ)

    tmpO = data["operations_per_product"]

    DUE_DATE_FLAG = False
    JOBS = []
    DUE_DATE = {}
    OPERATIONS = {}
    ELIGIBLE_MACHINES = {}
    OP_DURATIONS = {}

    TYPES = {}
    TYPES_LOOKUP = []
    SETUPS = {}
        
    for a in tmpJ:
        JOBS.append(a)
        if data["orders"][a]["due_date"] != None:
            DUE_DATE_FLAG = True
            # raise Exception("due date flag is true")
            DUE_DATE.setdefault(a,round(data["orders"][a]["due_date"] * 1000))
        else:
            DUE_DATE.setdefault(a,1000000) # dummy value
        
        tmpType = data["orders"][a]["product"]
        TYPES_LOOKUP.append(tmpType)
        ops = data["operations_per_product"][tmpType]
        OPERATIONS.setdefault(a,ops)
        
        for b in ops:
            ELIGIBLE_MACHINES.setdefault(a, {}).setdefault(b, data["operation_machine_compatibility"][tmpType][b])

            for c in ELIGIBLE_MACHINES[a][b]:
                tmpTime = round(dataS["processing_time"][a][b][c] * 1000) # TODO check if it still works
                OP_DURATIONS.setdefault(a, {}).setdefault(b, {}).setdefault(c, tmpTime)

    for m in MACHINES:
        TYPES.setdefault(m,[])
        for j in JOBS:
            tmpTO = JOBS.index(j) +1
            SETUPS.setdefault(m,{}).setdefault(tmpTO,[0 for i in range(NB_JOBS+1)])

    for a in tmpJ:
        tmpType = data["orders"][a]["product"]
        ops = data["operations_per_product"][tmpType]
    
        for b in ops:
            for c in ELIGIBLE_MACHINES[a][b]:

                tmpIndex = JOBS.index(a) +1
                TYPES[c].append(tmpIndex)
                for j in JOBS:
        
                    tmpFrom = JOBS.index(j) +1
                    jType = TYPES_LOOKUP[tmpFrom-1]
                    
                    SETUPS[c][tmpIndex][tmpFrom] = round(dataS["setup_time"][c][jType][a] * 1000) # TODO check if it still works

    for m in MACHINES:
        TYPES[m] = TYPES[m] + TYPES[m]

    #-----------------------------------------------------------------------------
    # Build the model
    #-----------------------------------------------------------------------------

    # Create model
    mdl = cp.CpoModel()

    # Create one interval variable per job, operation
    Z_io =  { (i,o) : cp.interval_var(name='Z_{}{}'.format(i,o))
            for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J])
            }

    # Create optional interval variable job operation machine if machine is eligible
    X_iom = { (i,o,MACHINES.index(M)) : cp.interval_var(name = 'X_{}{}{}'.format(i,o,MACHINES.index(M)), optional=True, size = OP_DURATIONS[J][O][M]) 
            for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J]) for M in ELIGIBLE_MACHINES[J][O]
            }

    # Variables to deal with setups
    S_io =  { (i,o) : cp.interval_var(name = 'S_{}{}'.format(i,o))
            for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J])
            }
    
    Sop_iom =   { 
                (i,o,MACHINES.index(M)) : cp.interval_var(name = 'Sop_{}{}{}'.format(i,o,MACHINES.index(M)), optional=True) 
                for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J]) for M in ELIGIBLE_MACHINES[J][O]
                }

    # Create sequence variable 
    MCH_m = { (m) : cp.sequence_var([X_iom[a] for a in X_iom if a[2] == m]
                                    + [Sop_iom[b] for b in Sop_iom if b[2] == m ], types = TYPES[M], name = 'MCH_{}'.format(m)) 
                for m,M in enumerate(MACHINES)
            }

    # ---------------------------------------------------------------------------------------------------
    # Constraints

    for j,J in enumerate(JOBS):
        for o,O in enumerate(OPERATIONS[J]):
            for M in ELIGIBLE_MACHINES[J][O]:
                m = MACHINES.index(M)

                mdl.add(cp.length_of(Sop_iom[j,o,m], 10000000) >= cp.element(SETUPS[M][j+1],cp.type_of_prev(MCH_m[m], Sop_iom[j,o,m], 0, 0)))


    # Force each operation to start after the end of the previous
    mdl.add(cp.end_before_start(Z_io[i,o-1], Z_io[i,o]) for i,o in Z_io if 0<o)

    # setups can only start when previous operation is finished! enforce it -> require S_iom again?
    mdl.add(cp.end_before_start(Z_io[i,o-1], S_io[i,o]) for i,o in S_io if 0<o)

    #setup before operation
    mdl.add(cp.end_at_start(Sop_iom[i,o,m], X_iom[i,o,m]) for i,o,m in Sop_iom)

    # Alternative constraints
    mdl.add(cp.alternative(Z_io[i,o], [X_iom[a] for a in X_iom if a[0:2]==(i,o)]) for i,o in Z_io)
    mdl.add(cp.alternative(S_io[i,o], [Sop_iom[a] for a in Sop_iom if a[0:2]==(i,o)]) for i,o in S_io)

    # Force no overlap for operations executed on a same machine
    mdl.add(cp.no_overlap(MCH_m[m]) for m in MCH_m)

    # add additional constraints same type with earlier due date needs to start before
    if DUE_DATE_FLAG:
        for i,i2 in enumerate(JOBS):
            for j,j2 in enumerate(JOBS):
                if TYPES_LOOKUP[i] == TYPES_LOOKUP[j]: 
                    if DUE_DATE[i2] < DUE_DATE[j2] or (DUE_DATE[i2] == DUE_DATE[j2] and i2 < j2):
                        mdl.add(cp.start_before_start(Z_io[i,0], Z_io[j,0]))

    # TODO additional constraints from  Yunusoglu (2022) paper?
    # TODO check model? improve solving? -> recent Cp model and settings! cite it as benchmark with PI!

    # objective criteria

    # specify Tardiness
    Tardiness = [cp.max(0, cp.end_of(Z_io[i, len(OPERATIONS[J]) -1]) - (DUE_DATE[J])) for i,J in enumerate(JOBS)]

    TotalTardiness = cp.sum(Tardiness)

    # specifiy flow time
    Flowtime = [cp.end_of(Z_io[i,len(OPERATIONS[J])-1]) - cp.start_of(Z_io[i,0]) for i,J in enumerate(JOBS)]

    TotalFlow = cp.sum(Flowtime)

    # specifiy makespan
    CMax = cp.max(cp.end_of(Z_io[i,o]) for i,o in Z_io)

    # number of tardy Jobs
    NB_Tardy = cp.sum(i != 0 for i in Tardiness)

    # Minimize objective!
    # create objective vector
    objVec = [CMax/1000, TotalTardiness/1000] # TODO same order as in agent! divide time based objectives by 10000
    # TODO use objective vector!
    mdl.add(cp.minimize(cp.scal_prod(objective,objVec)))
    # mdl.add(cp.minimize((CMax) / 60))

    #-----------------------------------------------------------------------------
    # Solve the model and display the result
    #-----------------------------------------------------------------------------

    # Solve model
    print('Solving model...')
    res = mdl.solve(TimeLimit= 600, # TODO dynamic stopping cirteria? and other params to tune solver!
                    agent = "local",
                    execfile="C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe")

    TmpObjective = res.get_objective_value()
    
    # TODO return solution as well?

    #-----------------------------------------------------------------------------
    # Show / Save Gantt
    #-----------------------------------------------------------------------------
    if saveGantt == True:
        colorMap = { p : ("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])) for p in tmpT }
        # Draw solution
        import docplex.cp.utils_visu as visu
        if res and visu.is_visu_enabled():
        # Draw solution
            visu.timeline('Solution for test')
            visu.panel('Machines')
            for m in range(NB_MACHINES):
                visu.sequence(name='RES_' + str(m+1))
                for a in X_iom:
                    if a[2]==m:
                        itv = res.get_var_solution(X_iom[a])
                        print(TYPES_LOOKUP[a[0]], ": ", itv)
                        if itv.is_present():
                            visu.interval(itv, colorMap[TYPES_LOOKUP[a[0]]], JOBS[a[0]])

                for b in Sop_iom:
                    if b[2]==m:
                        itv = res.get_var_solution(Sop_iom[b])
                        print(TYPES_LOOKUP[b[0]], ": ", itv)
                        if itv.is_present():
                            visu.interval(itv, colorMap[TYPES_LOOKUP[b[0]]], JOBS[b[0]])
            visu.show()
            #TODO save Gantt!


    return TmpObjective # TODO return solution as well?


def calculatePI(case, objective, sensitivity = "no"):

    """
    loop over all instances and according test samples in folder
    """

    if sensitivity == "setup":
        basedir = str(pathlib.Path(__file__).parent.resolve()) + "/Benchmark_Kurz/sensitivity_setupratio/data_" + case
    elif sensitivity == "uncertainty":
        basedir = str(pathlib.Path(__file__).parent.resolve()) + "/Benchmark_Kurz/sensitivity_uncertainty/data_" + case
    else:
        basedir = str(pathlib.Path(__file__).parent.resolve()) + "/Benchmark_Kurz/base_setting/data_" + case


    # get all files in directory excluding other directories
    # files = [f for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir,f))]
    
    # files = files[1:]
    files = ["lhmhlh-0_ii_Expo5.json"]

    # basedir = str(pathlib.Path(__file__).parent.resolve()) + "/Use_Case_Cereals/base_setting"
    # files = ["useCase_2_stages.json"]

    for filestr in files:
        with open(os.path.join(basedir,filestr)) as f:
            instance = json.load(f)

        sampledir = basedir + "/testing/" + filestr[0:-5]
        samples = [s for s in os.listdir(sampledir) if os.path.isfile(os.path.join(sampledir,s))]
        # samples = [s for s in samples if filestr[0:9] == s[0:9]]

        # print(samples)


        for s in samples:
            with open(os.path.join(sampledir,s)) as f2:
                sample = json.load(f2)

            # TODO only for testing!
            print("sample: ", s, "solution is:", CPsolution(instance, sample, objective, saveGantt = True))
            break

            if "objective_coupled" in sample.keys():
                continue
            else:
                # sample["objective_" + str(objective)] = CPsolution(instance, sample, objective)
                sample["objective_coupled"] = CPsolution(instance, sample, objective)

            with open(os.path.join(sampledir,s),'w') as f2:
                json.dump(sample, f2,indent=4)

def comparsolgantt(basedir, filestr,sample, objective):

    with open(os.path.join(basedir,filestr)) as f:
        instance = json.load(f)

    sampledir = basedir + "/testing/" + filestr[0:-5]
    with open(os.path.join(sampledir,sample)) as f2:
        sample = json.load(f2)

    CPsolution(instance, sample, objective, saveGantt = True)


    
# b = str(pathlib.Path(__file__).parent.resolve()) + "/Benchmark_Kurz/base_setting/data_ii"
# f = "llllll-0_ii_Expo5.json"
# s = "2.json"

# comparsolgantt(b,f,s,[1])

calculatePI("ii", [1,0], sensitivity = "no")
