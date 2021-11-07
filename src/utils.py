def convert_time(seconds):
    mins, sec = divmod(seconds, 60)
    hour, mins = divmod(mins, 60)
    if hour > 0:
        return "{:.0f} hour, {:.0f} minutes".format(hour, mins)
    elif mins > 5:
        return "{:.0f} minutes".format(mins)
    elif mins >= 2:
        return "{:.0f} minutes, {:.0f} seconds".format(mins, sec)
    elif mins > 0:
        return "{:.0f} minute, {:.0f} seconds".format(mins, sec)
    else:
        return "{:.2f} seconds".format(sec)
