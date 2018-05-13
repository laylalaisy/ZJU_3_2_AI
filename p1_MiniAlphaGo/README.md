# P1_MiniAlphaGo

特别对不起队友的一个作业，可以说是大学生涯最惨烈内疚的一个作业了。  
因为在学车每天感觉都要中暑而死，晕晕乎乎导致代码最后靠队友一个人全程carry我了。   
而报告竟然因为不是24：00之前而从床上惊吓下来糊了一个，可以说是大学最糟糕的一份report了。   


## 题目：
黑白棋(Reversi)，也叫苹果棋，翻转棋，是一个经典的策略性游戏。  
一般棋子双面为黑白两色，故称“黑白棋”。因为行棋之时将对方棋子翻转，变为己方棋子，故又称“翻转棋”(Reversi)。  
棋子双面为红、绿色的称为“苹果棋”。它使用8x8的棋盘,由两人执黑子和白子轮流下棋，最后子多方为胜方。   
随着网络的普及，黑白棋作为一种最适合在网上玩的棋类游戏正在逐渐流行起来。   
中国主要的黑白棋游戏站点有Yahoo游戏、中国游戏网、联众游戏等。   

### 游戏规则：
1．	棋局开始时黑棋位于e4和d5，白棋位于d4和e5，如图所示。   
2．	黑方先行，双方交替下棋。   
3．	一步合法的棋步包括：在一个空格新落下一个棋子，并且翻转对手一个或多个棋子。   
4．	新落下的棋子与棋盘上已有的同色棋子间，对方被夹住的所有棋子都要翻转过来。可以是横着夹，竖着夹，或是斜着夹。   
     夹住的位置上必须全部是对手的棋子，不能有空格。
5．	一步棋可以在数个（横向，纵向，对角线）方向上翻棋，任何被夹住的棋子都必须被翻转过来，棋手无权选择不去翻某个棋子。
6．	除非至少翻转了对手的一个棋子，否则就不能落子。如果一方没有合法棋步，也就是说不管他下到哪里，都不能至少翻转对手的一个棋子，   
     那他这一轮只能弃权，而由他的对手继续落子直到他有合法棋步可下。   
7．	如果一方至少有一步合法棋步可下，他就必须落子，不得弃权。   
8．	棋局持续下去，直到棋盘填满或者双方都无合法棋步可下。   
9．	如果某一方落子时间超过1分钟，则判该方失败。   

### 程序基本要求：
1.	以小组为单位完成，小组组员最多2人；   
2.	使用MCTS算法实现miniAlphaGo for Reversi；   
3.	MCTS算法部分需要自己实现，尽量不使用现成的包，工具或者接口；   
4.	在博弈过程中，miniAlphaGo每一步所花费的时间以及总时间需要显示出来；   
5.	需要有简单的图形界面用于人机博弈交互；   
6.	使用语言不限；   
7.	程序报告中需要说明每个人的分工与所完成的内容。   

### 实验Bonus   
1.	图形界面美观，可以参考已有软件实现相关功能；   
2.	如有创新或者独到之处可以在报告中指出。   


## Ref：
### Github：
https://github.com/kaychintam/MiniAlphaGo   
https://github.com/avartia/miniAlphaGo-for-Reversi   
### 蒙特卡洛树搜索MCTS：
https://zhuanlan.zhihu.com/p/25345778   
https://zhuanlan.zhihu.com/p/26335999   
https://zhuanlan.zhihu.com/p/34990220   
### TKinter:
http://effbot.org/tkinterbook/   
https://www.cnblogs.com/collectionne/p/6885066.html   
https://infohost.nmt.edu/tcc/help/pubs/tkinter/web/index.html   
http://www.runoob.com/python/python-gui-tkinter.html   
https://docs.python.org/2/library/tkinter.html   
### button:
https://infohost.nmt.edu/tcc/help/pubs/tkinter/web/button.html  


## Notes:
### MVC
MVC模式的意思是，软件可以分成三个部分。  
     • 视图（View）：用户界面。  
     • 控制器（Controller）：业务逻辑  
     • 模型（Model）：数据保存  
     
各部分之间的通信方式如下。   
     1. View 传送指令到 Controller  
     2. Controller 完成业务逻辑后，要求 Model 改变状态  
     3. Model 将新的数据发送到 View，用户得到反馈  
所有通信都是单向的。   
https://blog.csdn.net/tinym87/article/details/6957438


### event
对于想使用event的情况，像btn.bind("<Button-1>", handler)，又该怎么办呢，好说再写个中间适配器函数。event 会自动传到里面的函数   
```python
#coding=utf-8  
import Tkinter  
def handler(event, a, b, c):  
    '''''事件处理函数'''  
    print event  
    print "handler", a, b, c  
def handlerAdaptor(fun, **kwds):  
    '''''事件处理函数的适配器，相当于中介，那个event是从那里来的呢，我也纳闷，这也许就是python的伟大之处吧'''  
    return lambda event,fun=fun,kwds=kwds: fun(event, **kwds)  
if __name__=='__main__':  
    root = Tkinter.Tk()  
    btn = Tkinter.Button(text=u'按钮')  
# 通过中介函数handlerAdaptor进行事件绑定  
    btn.bind("<Button-1>", handlerAdaptor(handler, a=1, b=2, c=3))  
btn.pack()  
    root.mainloop()  
``` 
来自 <https://www.zhihu.com/question/42879591/answer/332029110>    


### canvas 
https://blog.csdn.net/pengzhi5966885/article/details/77774820 


### python对绑定事件的鼠标、按键的判断
当多个事件绑定了同一个命令，那么在命令内部根据不同的事件进行处理的时候，怎么确定哪个事件发生了呢，用下面的来检测，经过测试处理tab键和alt键不能识别，其他单个都能被识别。    
还有个事件的type属性，这个经过测试键盘事件返回字符2，鼠标返回字符2，可以根据这个再进行判断反会的是键盘事件还是鼠标事件。   
```python
# <Button-1>：鼠标左击事件
# <Button-2>：鼠标中击事件
# <Button-3>：鼠标右击事件
# <Double-Button-1>：双击事件
# <Triple-Button-1>：三击事件
from tkinter import *
tk = Tk()
canvas = Canvas(width=500,height=500)
canvas.pack()
#canvas.create_polygon(0,0,250,250,fill = 'red')
def echo_event(evt):
    #打印键盘事件
    if evt.type == "2":
        print("键盘：%s" % evt.keysym)
    #打印鼠标操作
    if evt.type == "4":
        print("鼠标： %s" % evt.num)
    #
    print(evt.type)
```
```python
	# 导入tkinter包，为其定义别名tk
	import tkinter as tk
	 
	# 定义Application类表示应用/窗口，继承Frame类
	class Application(tk.Frame):
	    # Application构造函数，master为窗口的父控件
	    def __init__(self, master=None):
	        # 初始化Application的Frame部分
	        tk.Frame.__init__(self, master)
	        # 显示窗口，并使用grid布局
	        self.grid()
	        # 创建控件
	        self.createWidgets()
	 
	    # 创建控件
	    def createWidgets(self):
	        # 创建一个文字为'Quit'，点击会退出的按钮
	        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
	        # 显示按钮，并使用grid布局
	        self.quitButton.grid()
	 
	# 创建一个Application对象app
	app = Application()
	# 设置窗口标题为'First Tkinter'
	app.master.title = 'First Tkinter'
	# 主循环开始
	app.mainloop()
```
来自 <https://www.cnblogs.com/collectionne/p/6885066.html>     


### 文件读取+写出
```python
 #dumps功能
 import pickle
 data = ['aa', 'bb', 'cc']  
 # dumps 将数据通过特殊的形式转换为只有python语言认识的字符串
 p_str = pickle.dumps(data)
 print(p_str)            
 b'\x80\x03]q\x00(X\x02\x00\x00\x00aaq\x01X\x02\x00\x00\x00bbq\x02X\x02\x00\x00\x00ccq\x03e.
 # loads功能
 # loads  将pickle数据转换为python的数据结构
 mes = pickle.loads(p_str)
 print(mes)
 ['aa', 'bb', 'cc']
 # dump功能
 # dump 将数据通过特殊的形式转换为只有python语言认识的字符串，并写入文件
 with open('D:/tmp.pk', 'w') as f:
     pickle.dump(data, f)
 # load功能
 # load 从数据文件中读取数据，并转换为python的数据结构
 with open('D:/tmp.pk', 'r') as f:
     data = pickle.load(f)
```
来自 <https://www.cnblogs.com/lincappu/p/8296078.html> 




