"""PRIMs model converted from task-switching.prims by pyprims.convert."""
from pyprims import Task

task = Task('task-switching',
            initial_skills=['choice-reaction'],
            lf=0.2,
            egs=0.05,
            rt=-1.0,
            input_activation=0.0,
            default_activation=1.0,
            alpha=0.05,
            goal_operator_learning=True,
            operator_bll=True,
            beta=0.05,
            reward=80.0,
            default_operator_assoc=8.0,
            default_inter_operator_assoc=4.0)

skill_choice_reaction = task.skill('choice-reaction')
skill_choice_reaction.operator('prepare-for-stimulus', '''
        V1 = fixation
        G2 = nil
        ==>
        *prepare -> G2
    ''')
skill_choice_reaction.operator('get-property', '''
        V1 = stimulus
        V2 = nil
        RT1 = nil
        ==>
        attend -> AC1
        *property -> AC2
    ''')
skill_choice_reaction.operator('get-mapping', '''
        V1 = stimulus
        V2 <> nil
        RT1 = nil
        ==>
        mapping -> RT1
        V2 -> RT2
    ''')
skill_choice_reaction.operator('press-key', '''
        RT1 = mapping
        ==>
        press -> AC1
        RT3 -> AC2
    ''')
skill_choice_reaction.operator('last-stimulus', '''
        V1 = end
        ==>
        say -> AC1
        stop -> AC2
        stop -> G1
    ''')

skill_prepare_next = task.skill('prepare-next')
skill_prepare_next.operator('set-first-task', '''
        V1 = fixation
        WM1 = nil
        ==>
        one -> WM1
        nil -> G2  
        wait -> AC1
    ''')
skill_prepare_next.operator('determine-next-task-retrieve-count', '''
        V1 = fixation
        RT1 = nil
        WM1 <> nil
        ==>
        count-fact->RT1
        WM1->RT2
    ''')
skill_prepare_next.operator('determine-next-task-decide-to-stay', '''
        RT1 = count-fact
        RT3 = two
        ==>
        RT3 -> WM1
        nil -> G2  // we stay with the old goal
        wait -> AC1
    ''')
skill_prepare_next.operator('determine-next-task-decide-to-switch', '''
        RT1 = count-fact
        RT3 = three
        ==>
        one -> WM1
        othertask -> RT1
        *property -> RT2
    ''')
skill_prepare_next.operator('set-other-task', '''
        RT1 = othertask
        ==>
        RT3 -> *property
        nil -> G2 
        wait -> AC1
    ''')

task.facts([
    ('cf1', 'count-fact', 'one', 'two'),
    ('cf2', 'count-fact', 'two', 'three'),
    ('ot1', 'othertask', 'size', 'food'),
    ('ot2', 'othertask', 'food', 'size'),
    ('mp1', 'mapping', 'fruit', 'left'),
    ('mp2', 'mapping', 'vegetable', 'right'),
    ('mp3', 'mapping', 'small', 'left'),
    ('mp4', 'mapping', 'large', 'right'),
])


@task.script
def script(m):
    foods = ["vegetable", "fruit"]
    sizes = ["large", "small"]
    m.instantiate_skill("choice-reaction", "food-task", ["prepare", "prepare-next"], ["property", "food"])
    m.instantiate_skill("choice-reaction", "size-task", ["prepare", "prepare-next"], ["property", "size"])
    m.add_binding("property", "food")
    m.add_binding("prepare", "prepare-next")
    i = 0
    while i <= 7:
        m.screen("fixation")
        m.run_relative_time(1.0)
        m.screen("stimulus")
        m.run_until_action("attend")
        action = m.last_action()
        choice = m.random(2)
        if action[0] == "attend":
            if action[1] == "food":
                m.print("Looking at food")
                cat = foods[choice]
                m.screen("stimulus", cat)
            if action[1] == "size":
                m.print("Looking at size")
                cat = sizes[choice]
                m.screen("stimulus", cat)
        m.run_until_action("press")
        i = i + 1
    m.screen("end")
    m.run_until_action("say")
    m.issue_reward()
    m.trial_end()
