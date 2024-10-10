import math


def twopoints_speed(id, x2, y2, w2, h2):
    speed = -1.0
    if id in twopoints_speed.tracks:
        x1, y1, w1, h1 = twopoints_speed.tracks[id]
        speed = twopoints_speed2(((x1, y1), (w1, h1)), ((x2, y2), (w2, h2)))
    
    twopoints_speed.tracks[id] = (x2, y2, w2, h2)
    
    return speed


twopoints_speed.tracks = {}

def twopoints_speed2(x1, x2):
    location1, location2 = x1[0], x2[0]
    (w1, h1) , (w2, h2) = x1[1], x2[1] 
    d_pixel = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    w = (w1 + w2) / 2
    h = (h1 + h2) / 2
    ix = w/h
    # mean w/h: 2.9
    ixp =  ix / 2.9 
    # defining thr pixels per meter
    # mean w: 4.2 m
    ppm = (w*ixp) / 4.2
    d_meters = d_pixel / ppm
    # fps: 30
    time_constant = 30 * 3.6
    #distance = speed/time
    speed = d_meters * time_constant
    print(w, h, ix, ixp, ppm, speed, d_pixel, location1, location2)
    return int(speed)


def twolines_speed(line1, line2):
    ppm = 8 
    d_pixel = line1[0][1] - line2[0][1]
    d_meters = d_pixel/ppm
    time_constant = 1/30
    #distance = speed/time
    speed = d_meters / time_constant *3.6

    return int(speed)

