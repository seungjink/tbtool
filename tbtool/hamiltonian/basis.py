import traceback
from collections import namedtuple

class Basis:
    """
    Set basis information during the calculation.
    
    :param ``quantumnumber``: Description of the parameter
    :type ``quantumnumber``: Union[str, list]

    The size should match to the that of quantumnumber.::

       basis = Basis(["spin", "orbital", "site"])
       basis = Basis("orbital")
    """
    def __init__(self, quantumnumber):
        try:
            iter(quantumnumber)
        except TypeError:
            quantumnumber = [quantumnumber]
        self._quantumnumber = quantumnumber
        self.basis = []
        self.state = namedtuple("state", self._quantumnumber)

    def addbasis(self, *args):
        """
        Add basis to current list.
        
        Parameters
        ----------
        args : list
            List of names of quantumnumbers


        The size should match to the that of quantumnumber.

        Example
        -------

        .. code-block:: python

            >>> bs = Basis(["spin", "orbital", "site"])
            >>> bs.addbasis(0, "px", (0,0,1))
            >>> bs.addbasis(1, "s", (1,1,1))
            >>> print(bs.basis)
            [state(spin=0, orbital='px', site=(0, 0, 1)), state(spin=1, orbital='s', site=(1, 1, 1))]

        """
        try:
            newbasis = self.state(*args)
        except TypeError:
            print(traceback.format_exc())
            exit("[Error] length of basis and args does not match.\n"
                 "Please check the number of arguments.")
        self.basis.append(newbasis)
    
    def getquantumnumber(self):
        """
        :return: Field name
        """
        return self.state._fields
    
    def getdimension(self):
        """
        :return: Current size of basis
        """
        return len(self.basis)

    def swap(self, i, j):
        """
        Exchange the order between two basis

        :param integer i: First index
        :param integer j: Second index
        
        """
        self.basis[i], self.basis[j] = self.basis[j], self.basis[i]
    
    def permute(self, permutation):

        """
        Exchange the order using permutation index.

        Args:
            permutation (list[int]): Permutation index

        Example:

            >>> b = Basis("orbitals")
            >>> b.addbasis("px")
            >>> b.addbasis("s")
            >>> b.addbasis("dxy")
            >>> print(b.basis)
            [state(orbitals='px'), state(orbitals='s'), state(orbitals='dxy')]
            >>> b.permute([0,2,1])
            >>> print(b.basis)
            [state(orbitals='px'), state(orbitals='dxy'), state(orbitals='s')]
        """
        if len(permutation) != self.getdimension():
            exit("[Error] The length of permutation matrix does not match\n"
                 "to your basis dimension")
        else:
            self.basis = [self.basis[i] for i in permutation]

bs = Basis(["spin", "orbital", "site"])
bs.addbasis(0, "px", (0,0,1))
bs.addbasis(1, "s", (1,1,1))
print(bs.basis)

bs = Basis("orbital")
bs.addbasis("dxy")
print(bs.basis)