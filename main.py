import pickle
from tkinter import Label, Text, END, Tk, Button

import numpy as np
import pandas as pd
import pymysql
tfid_model_file='E:\python代码\整合\output\model情感.pkl'
with open(tfid_model_file, 'rb') as infile:
    tfid_predict= pickle.load(infile)

dbconn = pymysql.connect(
    host="localhost",
    database="test",
    user="root",
    password="123456",
    port=3306,
    charset='utf8'
)
sqlcmd="select id,内容 from 评论"
myshuju=pd.read_sql(sqlcmd,dbconn)
#取前5行数据

b=myshuju.head()
print(myshuju)

root=Tk()
root.title("情感分析")
sw = root.winfo_screenwidth()
#得到屏幕宽度
sh = root.winfo_screenheight()
#得到屏幕高度
ww = 500
wh = 300
x = (sw-ww) / 2
y = (sh-wh) / 2-50
root.geometry("%dx%d+%d+%d" %(ww,wh,x,y))
# root.iconbitmap('tb.ico')
lb2=Label(root,text="输入内容，按回车键分析")
lb2.place(relx=0, rely=0.05)
txt = Text(root,font=("宋体",20))
txt.place(rely=0.7, relheight=0.3,relwidth=1)
inp1 = Text(root, height=15, width=65,font=("宋体",18))
inp1.place(relx=0, rely=0.2, relwidth=1, relheight=0.4)

def run1():
    txt.delete("0.0",END)
    a = inp1.get('0.0',(END))

    y = tfid_predict['lr'].predict(tfid_predict['tfidf'].transform([a]))
    if(y=='pos'):
        p = '好感'
    else:
        p= '反感'

    print(p)
    txt.insert(END, p)   # 追加显示运算结果

def button1(event):
    btn1 = Button(root, text='分析', font=("",12),command=run1) #鼠标响应
    btn1.place(relx=0.40, rely=0.6, relwidth=0.15, relheight=0.1)
    # inp1.bind("<Return>",run2) #键盘响应
haogan = tfid_predict['lr'].predict(tfid_predict['tfidf'].transform(myshuju['内容']))
pcount=np.sum((haogan=='pos'))
ncount=np.sum((haogan=='neg'))
print("好感和反感的比值为：",pcount/ncount)
button1(1)
root.mainloop()