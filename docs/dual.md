- `DualNumber` Module

    * Initialization

        Let's assume we have values `real` and `dual`, to create a dual object:

        ```python
            import AutoDiff.util import DualNumber

            dual_number = DualNumber(real, dual)  
        ```
    
    * Operators
    
        1. Arithmetic Operators
        
            The following arithmetic operators are supported with DualNumber:
            - `+`
            - `-`
            - `*`
            - `/` 
            - `exp()`
            
            Example:

            ```python
                n_one = DualNumber(real_one, dual_one) 
                n_two = DualNumber(real_two, dual_two)

                add_result = n_one + n_two
                minus_result = n_one - n_two
                multiply_result = n_one * n_two
                division_result = n_one / n_two
                Exp_result = n_one.exp(2)
            ```
    
        2. Trigonometry Operators

            The following trigonometry operators are supported with DualNumber:
            - `sin()`
            - `cos()`
            - `tan()`
            
            Example:

            ```python
                import AutoDiff as ad
                n_one = DualNumber(real_one, dual_one) 

                sin_result = ad.sin(n_one)
                cos_result = ad.cos(n_one)
                tan_result = ad.tan(n_one)
            ```