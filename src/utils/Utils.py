import os
import numpy as np


def save_answer(task_number, answers, line=False, space=False):
    """
    Method prints and saves an answer on a task
    
    :param task_number: the task number, also uses like a file name
    :param answers:     the answers
    :param space:       use a space after each answer
    :param line:        use a line terminator after each answer
    :return:            none
    """
    # Convert to string
    task_number = str(task_number)

    # Print the answer
    print()
    print('Task №' + task_number + ' answer ' + str(answers))
    print()

    # Create folder if not exists
    path = 'answers'
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + '/' + task_number + '.txt', 'w+') as file:

        # If it isn't an array make it like an array
        # noinspection PyUnresolvedReferences
        if not isinstance(answers, (list, np.ndarray)):
            file.write(str(answers))
        else:
            if line:
                # noinspection PyTypeChecker
                answers = [str(a) + '\n' if idx != len(answers) - 1 else str(a) for idx, a in enumerate(answers)]
                file.writelines(answers)
            if space:
                # noinspection PyTypeChecker
                answers = [str(a) for a in answers]
                file.write(' '.join(answers))
