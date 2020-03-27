# %%

class Book():
    
    name = 'tess'
    
    def __init__(self):
        self.name = self.name + ' coba'

    def __init__(self, *args, **kwargs):
        self.name = self.name + ' coba'

    # def __init__(self,name, *args, **kwargs):
    #     self.name = name + ' coba'

    def printself(self):
        print('name printself ',self.name)
    def printnoself(self):       
        print('name printnoself ',self.name)
    def printnopar(self,name,*args, **kwargs):
        # name = 'localnopar'        
        print('name printnopar ',name)
        print('args ',args)
        print('kwargs ',kwargs)

def printnoselfoutside():
        name = 'local'
        print('name printnoself '+name)

# def functiontest(self, *args, **kwargs):
def functiontest(book,*args, **kwargs):
#def functiontest():
    if book:
        print('yes')
    # pass

if __name__ == "__main__":
    # main()
    # functiontest()
    book = Book()
    functiontest(book)
    book2 = Book('anu')
    book.printself()
    book2.__init__()
    book2.printself()
    book.printnoself()
    book.printnopar('aaa')
    book.printnopar('bbb','sddd',buku='ccc')
    # book.printnopar()
    # book.printnoself('ass')

# %%

# %%
