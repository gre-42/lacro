#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('text', nargs='?', default='')
    parser.add_argument('--ord', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--infile')
    parser.add_argument('--latex_outfile')
    parser.add_argument('--outfile')
    parser.add_argument('--horizontal', action='store_true')
    parser.add_argument('--swap', action='store_true')
    parser.add_argument('--print_symbols', action='store_true',
                        help='Print symbols and exit')

    return parser.parse_args()


def run(args):
    from lacro.io.string import load_string_from_file, save_string_to_file

    if args.print_symbols:
        from lacro.collections import items_2_str
        from lacro.parse.latex2unicode import symbols
        print(items_2_str(symbols.items(), width='auto', colon=False,
                          isstr=True))

    def to_latex(s):
        from lacro.parse.latex2unicode import latex2unicode, unicode2latex
        if args.invert:
            res = unicode2latex(s)
        else:
            res = latex2unicode(s)
        return ' '.join('%x' % ord(v) for v in res) if args.ord else res

    contents = args.text + ('' if args.infile is None else
                            load_string_from_file(args.infile))

    if not args.interactive:
        print(to_latex(contents))
        code = 0
    else:
        import curses

        from lacro.string.textbox import InsertingTextbox, fill_window

        textbox = None
        try:
            debug = False

            def main(mainwindow):
                nonlocal textbox
                # Clear screen
                # mainwindow.clear()

                # This raises ZeroDivisionError when i == 10.
                # for i in range(1, 11):
                # v = i
                # mainwindow.addstr(i, 0,
                #                   '10 divided by {} is {}'.format(v, 10/v))

                # mainwindow.refresh()
                # mainwindow.getkey()

                # curses.cbreak()
                # mainwindow.keypad(1)
                # curses.noecho()

                maxy, maxx = mainwindow.getmaxyx()
                if args.horizontal:
                    x0 = (maxx + 1) // 2
                    x1 = maxx // 2
                    o_args = (maxy, x1, 0, x0)
                    e_args = (maxy, x0)
                else:
                    y0 = (maxy + 1) // 2
                    y1 = maxy // 2
                    o_args = (y1, maxx, y0, 0)
                    e_args = (y0, maxx)

                out_win  = curses.newwin(*(e_args if args.swap else o_args))
                edit_win = curses.newwin(*(o_args if args.swap else e_args))

                # out_win.box()
                # edit_win.box()

                textbox = InsertingTextbox(edit_win, out_win, verbose=debug,
                                           contents=contents)

                # out_win.refresh()
                # edit_win.refresh()

                def onchange(current_input):
                    try:
                        res = to_latex(current_input)
                    except Exception as e:
                        res = 'Error: %s' % e
                    if not debug:
                        out_win.clear()
                        fill_window(out_win, res)

                    out_win.refresh()
                    # edit_win.refresh()
                textbox.edit(onchange)
                # from lacro.io.textInput import BasicValidator
                # vali = BasicValidator()
                # textbox.edit(lambda key: vali.validate(key, textbox))

                # edit_win.refresh()
                # out_win.refresh()
                # mainwindow.refresh()
                # edit_win.getkey()
                # edit_win.getkey()

            curses.wrapper(main)
            code = 0
        except KeyboardInterrupt:
            code = 1
        contents = textbox.contents.text

    if args.outfile is not None:
        save_string_to_file(args.outfile, to_latex(contents))
    if args.latex_outfile is not None:
        save_string_to_file(args.latex_outfile, contents)
    import sys
    sys.exit(code)


def main():
    run(parse_args())
