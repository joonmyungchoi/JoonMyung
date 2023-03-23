from flask import Flask, request, render_template
import queue
import threading
import subprocess
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
process_queue = queue.PriorityQueue()
next_pidx = 0

@app.route('/')
def home():
    return '''
        <!DOCTYPE html>
        <html>
            <head>
                <title>Process Queue</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                    }
                    h1 {
                        text-align: center;
                    }
                    form {
                        display: flex;
                        flex-direction: column;
                        width: 50%;
                        margin: 0 auto;
                    }
                    label {
                        margin-bottom: 0.5rem;
                    }
                    input[type="text"],
                    input[type="number"] {
                        padding: 0.5rem;
                        border-radius: 4px;
                        border: 1px solid #ccc;
                        margin-bottom: 1rem;
                    }
                    button[type="submit"] {
                        padding: 0.5rem;
                        border-radius: 4px;
                        background-color: #4CAF50;
                        color: #fff;
                        border: none;
                        cursor: pointer;
                    }
                    button[type="submit"]:hover {
                        background-color: #3e8e41;
                    }
                    a {
                        display: block;
                        margin-top: 2rem;
                        text-align: center;
                    }
                    ul {
                        list-style: none;
                        padding-left: 0;
                    }
                    li {
                        margin-bottom: 1rem;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }
                    li button[type="submit"] {
                        background-color: #f44336;
                    }
                    li button[type="submit"]:hover {
                        background-color: #c62828;
                    }
                    li form:first-child {
                        margin-right: 1rem;
                    }
                    form:last-of-type {
                        margin-top: 2rem;
                    }
                </style>
            </head>
            <body>
                <h1>Process Queue</h1>
                <form method="POST" action="/enqueue">
                    <label for="process">Process:</label>
                    <input type="text" id="process" name="process" required>
                    <label for="priority">Priority:</label>
                    <input type="number" id="priority" name="priority" required>
                    <button type="submit">Enqueue</button>
                </form>
                <a href="/queue">View Queue</a>
            </body>
        </html>
    '''


# from flask import Flask, request
# import queue
# import threading
# import subprocess

# app = Flask(__name__)
# process_queue = queue.PriorityQueue()
# next_pidx = 0

# @app.route('/')
# def home():
#     return '''
#         <form method="POST" action="/enqueue">
#             <label for="process">Process:</label>
#             <input type="text" id="process" name="process" required>
#             <label for="priority">Priority:</label>
#             <input type="number" id="priority" name="priority" required>
#             <button type="submit">Enqueue</button>
#         </form>
#         <a href="/queue">View Queue</a>
#     '''

@app.route('/enqueue', methods=['POST'])
def enqueue():
    global next_pidx
    process = request.form['process']
    priority = int(request.form['priority'])
    process_queue.put((priority, process, next_pidx))
    next_pidx += 1
    return 'Process enqueued.'

@app.route('/remove', methods=['POST'])
def remove():
    pidx_to_remove = request.form['process_to_remove']
    for priority, process, pidx in list(process_queue.queue):
        if pidx == int(pidx_to_remove):
            process_queue.queue.remove((priority, process, pidx))
    return 'Process removed.'

@app.route('/change_priority', methods=['POST'])
def change_priority():
    pidx_to_change = request.form['process_to_change']
    new_priority = int(request.form['new_priority'])
    for priority, process, pidx in list(process_queue.queue):
        if pidx == int(pidx_to_change):
            process_queue.queue.remove((priority, process, pidx))
            process_queue.put((new_priority, process, pidx))
    return 'Process priority changed.'

# @app.route('/queue')
# def view_queue():
#     items = list(process_queue.queue)
#     if not items:
#         return 'Queue is empty.'
#     else:
#         items_html = '<ul>'
#         for priority, process, pidx in items:
#             items_html += f'''
#                 <li>
#                     {process} (priority: {priority}, process index : {pidx})
#                     <form method="POST" action="/remove" style="display: inline-block;">
#                         <input type="hidden" name="process_to_remove" value="{pidx}">
#                         <button type="submit">Remove</button>
#                     </form>
#                     <form method="POST" action="/change_priority" style="display: inline-block;">
#                         <input type="hidden" name="process_to_change" value="{pidx}">
#                         <label for="new_priority">New priority:</label>
#                         <input type="number" id="new_priority" name="new_priority" required>
#                         <button type="submit">Change priority</button>
#                     </form>
#                 </li>
#             '''
#         items_html += '</ul>'
#         return items_html


@app.route('/queue')
def view_queue():
    items = list(process_queue.queue)
    return render_template('queue.html', items=items)


def execute_processes():
    while True:
        priority, process, pidx = process_queue.get()
        print(f'Executing process: {process} with process index : {pidx}')
        subprocess.run(process, shell=True, stdout=subprocess.DEVNULL)
        process_queue.task_done()
        print(f'{process} done.')

if __name__ == '__main__':
    execution_thread = threading.Thread(target=execute_processes)
    execution_thread.start()
    app.run(host='0.0.0.0', port='5000')
