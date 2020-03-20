import tkinter as tk

def userInput():
    weightList = []

    window = tk.Tk()

    label = tk.Label(
        master = window,
        text = 'Enter a weight')
    label.pack()

    entry = tk.Entry(
        master = window)
    entry.pack()

    def collectData():
        weight = entry.get()
        print(weight)
        weightList.append(weight)

    button = tk.Button(
        master = window,
        text = 'Update Data',
        command = collectData)
    button.pack()

    window.mainloop()
