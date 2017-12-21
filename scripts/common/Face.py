

class Face():

    def __init__(self,top,left,bottom,right,label):
        self.top=top
        self.bottom=bottom
        self.right=right
        self.left=left
        self.label=label
        self.size=str(pow(bottom-top,2)+pow(right-left,2))
    def encode(self,code):
        self.encoding=code
        print'ee'
        

        
        

