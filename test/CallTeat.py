#class中的下划线方法
class person:
    def __call__(self, name):#下划线表示内置函数
        print("call"+"hello:",name)

    def hello(self,name):
        print("hello",name)

person=person()
person("amy")#调用了__call__方法
person.hello("amy")#调用了hello方法