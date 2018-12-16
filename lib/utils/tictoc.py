import time

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    return startTime_for_tictoc

def toc():
    if 'startTime_for_tictoc' in globals():
        endTime = time.time()
        return endTime - startTime_for_tictoc
    else:
        return None