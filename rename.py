import os
  

  
for count, filename in enumerate(os.listdir("E:\FYP\Final pipeline\Dataset\Dressing Table\\")):
    dst ="D" + str(count) + ".jpg"
    src ='E:\FYP\Final pipeline\Dataset\Dressing Table\\'+ filename
    dst ='E:\FYP\Final pipeline\Dataset\Dressing Table\\'+ dst
        
    # rename() function will
    # rename all the files
    os.rename(src, dst)
