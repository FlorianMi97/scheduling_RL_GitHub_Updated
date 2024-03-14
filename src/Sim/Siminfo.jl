Base.@kwdef mutable struct Siminfo
    # TODO add more if needed????
    nextevent # format (time,flag)
    activitytimes   # for each machine this stores the current activity:
                    # the minimum the activity in (absolut setup, realtiv process and the absolute process threshold)
                    # the mean time of the activities in relativ values (setup, process)
                    # the real value of the activity in absolut values (setup, process)
                    # the distribution type of the activities (setup, process)
                    # the max value of the activity if defined (absolute setup, relativ process)
    pointercurrentop # pointer of job to its current operation!
    pointersample # TODO point to current use env.sample or nothing if sample is not used
    notfinished # list of jobnumbers not finish?

    # metrics
    time # makespan
    flowtime
    tardiness
    discretetardiness # TODO truck once a day
    numtardyjobs
    numsetups
    
    mappingjobs
    mappingmachines

    edd_type_machine
end

function initsiminfo(p::Problem, am)
    ne = [[0.0,0] for _ in p.machines]
    at = [[[0.0, 0.0, 0.0],[0.0, 0.0],[0.0, 0.0], ["None", "None"], [0.0, 0.0]] for _ in p.machines]
    mj = p.mapindexorder #immutable from problem
    mm = p.mapindexmachine # immutable from problem

    if "dds" ∈ am

        edd_type_machine = Dict(type => Dict(machine => Dict("scheduling" => Inf, "waiting" => Inf) for machine in p.machines) for type in keys(p.mapindexprods))
        for j ∈ 1:length(p.orders)
            o = p.orders[j]
            for m in o.eligmachines[o.ops[1]]
                if o.duedate < edd_type_machine[o.type][m]["scheduling"]
                    edd_type_machine[o.type][m]["scheduling"] = o.duedate
                    edd_type_machine[o.type][m]["waiting"] = o.duedate
                end
            end
        end
    else
        edd_type_machine = nothing
    end

    Siminfo(    nextevent = ne, pointercurrentop = Dict(j => 1 for j in keys(p.orders)), pointersample = nothing,
                activitytimes = at,notfinished = sort(collect(keys(mj))),time = 0.0, flowtime = Dict(j => [0.0,nothing] for j in keys(p.orders)),
                tardiness = Dict(j => 0.0 for j in keys(p.orders)), discretetardiness = Dict(j => 0.0 for j in keys(p.orders)),
                numtardyjobs = 0, numsetups = 0, mappingjobs = mj , mappingmachines = mm, edd_type_machine = edd_type_machine
            )
end

# function resetsiminfo!(s::Siminfo)
#     s.nextevent = [[0.0,0] for _ in s.nextevent]
#     s.activitytimes = [[[0.0, 0.0, 0.0],[0.0, 0.0],[0.0, 0.0]] for _ in s.activitytimes]
#     s.pointercurrentop = Dict(x => 1 for x in keys(s.pointercurrenttop))
#     s.notfinished = sort(collect(keys(s.mappingjobs)))

#     # metrics
#     s.time = 0.0
#     s.flowtime = Dict(x => [0.0,false] for x in keys(s.flowtime))
#     s.tardiness = Dict(x => 0.0 for x in keys(s.tardiness))
#     s.discretetardiness = Dict(x => 0.0 for x in keys(s.discretetardiness))# TODO truck once a day
#     s.numtardyjobs = 0
#     s.numsetups = 0
# end

function setsamplepointer!(s::Siminfo,p)
    s.pointersample = p
end

function update_edd!(env::AbstractEnv)
    for m ∈ env.problem.machines
        DDS = Dict(prod => Inf for prod ∈ keys(env.problem.mapindexprods))
        DDW = Dict(prod => Inf for prod ∈ keys(env.problem.mapindexprods))
        for j ∈ 1:length(env.problem.orders)
            if env.siminfo.pointercurrentop[j] != 0
                o = env.problem.orders[j]
                if m ∈ o.eligmachines[o.ops[env.siminfo.pointercurrentop[j]]]
                    if env.state.executable[j]
                        if o.duedate < DDS[o.type]
                            DDS[o.type] = o.duedate
                        end
                    else
                        if o.duedate < DDW[o.type]
                            DDW[o.type] = o.duedate
                        end
                    end
                end
            end

        end
        for prod ∈ keys(env.problem.mapindexprods)
            env.siminfo.edd_type_machine[prod][m]["scheduling"] = DDS[prod]
            env.siminfo.edd_type_machine[prod][m]["waiting"] = DDW[prod]
        end
    end
end