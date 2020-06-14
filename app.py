from tkinter import *
import pandas as pd
from tkinter import filedialog
from tkinter import messagebox

from main import forward_selection, Backward_Elimination, Stepwise


class Application(Frame):
    def loadDataFromCsv(self):
        try:
            file_name = filedialog.askopenfilename(
                filetypes=(("сsv files", "*.csv"),)
            )
            self.data = pd.read_csv(
                file_name,
                sep=';',
                decimal='.'
            )
            features = self.data.columns.values
            self.listbox1.delete(0, END)
            for i in features:
                self.listbox1.insert(END, i)

        except FileNotFoundError:
            messagebox.showerror("Error", "Indicate file")

    def regression(self):
        global result
        target = self.listbox1.get(ACTIVE)
        if self.method_type.get() == 1:
            print('Прямой отбор')
            result = forward_selection(0.05, self.data, target)
        elif self.method_type.get() == 2:
            print('Обратное исключение')
            result = Backward_Elimination(0.05, self.data, target)
        elif self.method_type.get() == 3:
            print('Включение и исключение')
            result = Stepwise(0.05, self.data, target)

        print(result)

        if len(result) > 0:
            self.result.set(', '.join(map(str, result)))
        else:
            self.result.set("Нет значимых переменных")

    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        mainmenu = Menu(root)
        root.config(menu=mainmenu)
        filemenu = Menu(mainmenu, tearoff=0)
        filemenu.add_command(label="Открыть...", command=self.loadDataFromCsv)
        mainmenu.add_cascade(label="Файл", menu=filemenu)

        self.result = StringVar()
        self.result.set("")

        Label(root, text="Отобранные переменные:") \
            .grid(row=0, column=0)

        self.resultLabel = Label(root, textvariable=self.result) \
            .grid(row=1, column=0, sticky=N)

        Label(root, text="Целевая переменная: ").grid(row=0, column=4, sticky=N)
        self.listbox1 = Listbox(
            root,
            selectmode=SINGLE
        )
        self.listbox1.grid(row=1, column=4, sticky=N)

        Label(root, text="Метод построения модели: ").grid(row=3, column=4, sticky=N)

        self.method_type = IntVar()
        self.method_type.set(1)
        Radiobutton(
            text='Прямой отбор',
            value=1,
            variable=self.method_type,
            padx=15,
            pady=10
        ).grid(row=4, column=4, sticky=W)

        Radiobutton(
            text="Обратное исключение",
            value=2,
            variable=self.method_type,
            padx=15,
            pady=10
        ).grid(row=5, column=4, sticky=W)

        Radiobutton(
            text="Включение и исключение",
            value=3,
            variable=self.method_type,
            padx=15,
            pady=10
        ).grid(row=6, column=4, sticky=W)

        Button(root, text="Построить модель", command=self.regression) \
            .grid(row=7, column=4)

    def say_hi(self):
        print("hi there, everyone!")


root = Tk()
app = Application(master=root)
app.mainloop()
