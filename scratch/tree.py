'''
Created on 12.01.2016

@author: Yingxiong
'''
from traits.api \
    import HasTraits, Str, Regex, List, Instance, Float, Property
from traitsui.api \
    import TreeEditor, TreeNode, View, Item, VSplit, \
    HGroup, Handler, Group
from traitsui.menu \
    import Menu, Action, Separator
from traitsui.wx.tree_editor \
    import NewAction, CopyAction, CutAction, \
    PasteAction, DeleteAction, RenameAction


class Chapter(HasTraits):

    number_of_pages = Float
    name = Str


class Section(HasTraits):

    chapter = List(Chapter)


class Content(HasTraits):

    nop = Float(1)


class Text(HasTraits):

    section = List(Section)

#     content = List(Content)
    content = Property

    def _get_content(self):
        return [Content()]

    name = Str('book')


class Book(HasTraits):

    text = Instance(Text)

    tree_editor = TreeEditor(nodes=[TreeNode(node_for=[Text],
                                             children='',
                                             label='name'),
                                    TreeNode(node_for=[Text],
                                             children='content',
                                             label='=Content'),
                                    TreeNode(node_for=[Content],
                                             children='',
                                             label='=Content'),
                                    TreeNode(node_for=[Text],
                                             children='section',
                                             label='=Section'),
                                    TreeNode(node_for=[Section],
                                             children='chapter',
                                             label='=Chapters',
                                             view=View()),
                                    TreeNode(node_for=[Chapter],
                                             children='',
                                             label='name')],
                             orientation='vertical')

    view = View(Group(Item('text', editor=tree_editor)))

if __name__ == '__main__':

    ch1 = Chapter(number_of_pages=10., name='ch 1')
    ch2 = Chapter(number_of_pages=20., name='ch 2')
    ch3 = Chapter(number_of_pages=30., name='ch 3')

    sec1 = Section(chapter=[ch1, ch2, ch3])

    ch4 = Chapter(number_of_pages=10., name='ch 4')
    ch5 = Chapter(number_of_pages=20., name='ch 5')

    sec2 = Section(chapter=[ch4, ch5])

    con = Content()
    text = Text(section=[sec1, sec2])

    book = Book(text=text)

    book.configure_traits()
