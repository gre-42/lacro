# -*- coding: utf-8 -*-
import curses
from curses import textpad

import lacro.stdext as se


class Contents:

    def __init__(self, text):
        self.text = text

    def yx_2_i(self, y, x):
        # len(l)+1 because newline character is removed
        return x + sum(len(l) + 1 for i, l in enumerate(self.text.split('\n')) if i < y)

    def i_2_yx(self, i):
        y, x = se.last_element((y, len(l)) for y, l in enumerate(self.text[:i].split('\n')))
        return y, x

    def insert(self, y, x, char):
        i = self.yx_2_i(y, x)
        self.text = self.text[:i] + char + self.text[i:]
        return self.i_2_yx(i + len(char))

    def delete_before(self, y, x, num):
        i = self.yx_2_i(y, x)
        num = min(num, i)
        self.text = self.text[:i - num] + self.text[i:]
        return self.i_2_yx(i - num)

    def delete_after(self, y, x, num):
        i = self.yx_2_i(y, x)
        self.text = self.text[:i] + self.text[i + num:]

    def clip(self, maxy, maxx):
        self.text = '\n'.join(l[:maxx] for l in self.text.split('\n')[:maxy])
        return self

    def clip_win(self, window):
        maxy, maxx = window.getmaxyx()
        self.clip(maxy, maxx - 1)

    def bound_yx(self, y, x):
        y = min(max(0, y), len(self.text.split('\n')) - 1)
        x = min(max(0, x), len(self.text.split('\n')[y]))
        return y, x


def fill_window(window, text):
    cont = Contents(text)
    cont.clip_win(window)
    #se.save_string_to_file('text.txt', cont.text)
    #se.save_string_to_file('size.txt', str(window.getmaxyx()))
    window.addstr(0, 0, cont.text)


class InsertingTextbox:

    def __init__(self, window, debugwin, verbose=False, contents=''):
        # if contents != '':
                    #fill_window(window, contents)
        self.window = window
        self.textbox = textpad.Textbox(window)
        #self.textbox.stripspaces = False
        self.debugwin = debugwin
        self.contents = Contents(contents)
        self.verbose = verbose
        # if contents != '':
        #fill_window(self.window, self.contents.text)

    # def set_text(self, contents):
        #y, x = self.window.getyx()
        # self.window.clear()
        #self.window.addstr(0, 0, contents)
        #self.contents = Contents()
        #self.window.move(y, x)

    def edit(self, onchange):
        def validator(key):
                    # https://lists.torproject.org/pipermail/tor-commits/2011-September/035508.html
            y, x = self.window.getyx()
            y0, x0 = y, x
            #se.save_string_to_file('asd.txt', str(key))
            #current_input = self.textbox.gather()
            if (key == 10) or curses.ascii.isprint(key):
                # Shifts the existing text forward so input is an insert method rather
                # than replacement. The curses.textpad accepts an insert mode flag but
                # this has a couple issues...
                # - The flag is only available for Python 2.6+, before that the
                #   constructor only accepted a subwindow argument as per:
                #   https://trac.torproject.org/projects/tor/ticket/2354
                # - The textpad doesn't shift text that has text attributes. This is
                #   because keycodes read by self.window.inch() includes formatting,
                #   causing the curses.ascii.isprint() check it does to fail.
                #lines = current_input.split('\n')
                # if y < len(lines):
                            #self.window.addstr(y, x + 1, lines[y][x:self.textbox.maxx - 1])
                y, x = self.contents.insert(y, x, chr(key))
                #self.contents.clip(self.textbox.maxy, self.textbox.maxx)
                self.contents.clip_win(self.window)
            elif key == 27:
                # curses.ascii.BEL is a character codes that causes textpad to terminate
                return curses.ascii.BEL
            elif key == curses.KEY_HOME:
                self.window.move(y, 0)
                return None
            elif key == curses.KEY_END:
                self.window.move(*self.contents.bound_yx(y, 99999))
                return None
            elif key == curses.KEY_RIGHT:
                self.window.move(*self.contents.bound_yx(y, x + 1))
                return None
            elif key == curses.KEY_LEFT:
                self.window.move(*self.contents.bound_yx(y, x - 1))
                return None
            elif key == curses.KEY_UP:
                self.window.move(*self.contents.bound_yx(y - 1, x))
                return None
            elif key == curses.KEY_DOWN:
                self.window.move(*self.contents.bound_yx(y + 1, x))
                return None
            elif key == curses.KEY_BACKSPACE:
                y, x = self.contents.delete_before(y, x, 1)
                self.contents.clip_win(self.window)
            elif key == curses.KEY_DC:
                self.contents.delete_after(y, x, 1)
                self.contents.clip_win(self.window)
            elif key == 410:
                # if we're resizing the display during text entry then cancel it
                # (otherwise the input field is filled with nonprintable characters)
                return curses.ascii.BEL
            elif key == 0:
                pass
            else:
                if self.verbose:
                    self.debugwin.clear()
                    fill_window(self.debugwin, 'Unknown character: %d' % key)
                return None
            if self.verbose:
                self.debugwin.clear()
                fill_window(self.debugwin, 'x0 %d y0 %d x %d y %d\n' % (x0, y0, x, y) + se.added_line_numbers(self.contents.text))

            self.window.clear()
            fill_window(self.window, self.contents.text)
            y, x = self.contents.bound_yx(y, x)
            self.window.move(y, x)
            # for i,l in enumerate(se.added_line_numbers(self.contents).split('\n')):
            #se.save_string_to_file('asd.txt', l.strip())
            #self.debugwin.addstr(2+i, 0, l)
            #self.debugwin.addstr(2+i, 0, 'asd\n\nihh')
            #se.save_string_to_file('asd.txt', str(key))

            onchange(self.contents.text)

            return None
        if self.contents != '':
            validator(0)
        self.textbox.edit(validator)
        # while True: # does not work with up/down keys because they have values > 256
        # validator(self.window.getch())
