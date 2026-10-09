"""PRIMs model converted from stroop.prims by pyprims.convert."""
from pyprims import Task

task = Task('stroop',
            initial_skills=['stroop'],
            lf=0.2,
            egs=0.05,
            rt=-3.0,
            alpha=0.05,
            default_activation=1.0,
            input_activation=1.0,
            default_operator_assoc=8.0,
            default_inter_operator_assoc=4.0,
            goal_operator_learning=True,
            operator_bll=True,
            beta=0.05,
            reward=10.0)

skill_stroop = task.skill('stroop')
skill_stroop.operator('retrieve-response', '''
        V2 <> nil
        RT1 = nil
        ==>
        response -> RT1
        V2 -> RT2
    ''')
skill_stroop.operator('say-color', '''
        RT3 <> nil
        ==>
        say -> AC1
        RT3 -> AC2
        stop -> G1
    ''')

skill_default_attend = task.skill('default-attend')
skill_default_attend.operator('just-wait', '''
        V1 = fixation
        ==>
        wait -> AC1
    ''', activation=1.5)
skill_default_attend.operator('attend', '''
        V1 = stim
        V2 = nil
        ==>
        attend -> AC1
    ''', activation=1.5)

skill_attend_color = task.skill('attend-color')
skill_attend_color.operator('prepare-for-stimulus', '''
        V1 = fixation
        G2 = nil
        ==>
        *prepare -> G2
    ''')
skill_attend_color.operator('wait-when-prepared', '''
        V1 = fixation
        G2 <> nil
        ==>
        wait -> AC1
    ''')

skill_focus_color = task.skill('focus-color')
skill_focus_color.operator('attend-just-color', '''
        V1 = stim
        V2 = nil
        ==>
        attend-color -> AC1
    ''')

task.action('say', latency=0.3, noise=0.1, distribution='uniform', output='Saying')
task.action('attend', latency=0.085, output='Attending')
task.action('attendcolor', latency=0.085, output='AttendingColor')
task.facts([
    ('red-word-assoc', 'response', 'red-word', 'red', ':activation', '0.0'),
    ('blue-word-assoc', 'response', 'blue-word', 'blue', ':activation', '0.0'),
    ('red-color-assoc', 'response', 'red-color', 'red', ':activation', '0.0'),
    ('blue-color-assoc', 'response', 'blue-color', 'blue', ':activation', '0.0'),
])

task.sji('red-word', 'red-word-assoc', 1.5)
task.sji('red-word', 'blue-word-assoc', -1.5)
task.sji('red-color', 'blue-color-assoc', -1.5)
task.sji('red-color', 'red-color-assoc', 1.5)
task.sji('blue-word', 'red-word-assoc', -1.5)
task.sji('blue-word', 'blue-word-assoc', 1.5)
task.sji('blue-word', 'red-color-assoc', -1.5)
task.sji('blue-word', 'blue-color-assoc', 1.5)
task.sji('red-word', 'blue-color-assoc', -1.5)
task.sji('red-word', 'red-color-assoc', 1.5)
task.sji('blue-color', 'blue-color-assoc', 1.5)
task.sji('blue-color', 'red-color-assoc', -1.5)

@task.script
def script(m):
    trials = [["red-color", "red-word", "consistent"], ["blue-color", "blue-word", "consistent"], ["red-color", "blue-word", "conflict"], ["blue-color", "red-word", "conflict"]]
    m.add_binding("prepare", "focus-color")
    for trial in trials:
        m.set_data_file_field(1, trial[2])
        m.print("Presenting", trial[2], "with", trial[0], "and", trial[1])
        m.screen("fixation")
        m.run_relative_time(1.0)
        m.screen("stim")
        m.run_until_action()
        ac = m.last_action()
        if ac[0] == "attend":
            m.screen("stim", trial[0], trial[1])
        else:
            m.screen("stim", trial[0])
        m.run_until_action("say")
        m.issue_reward()
        m.trial_end()
