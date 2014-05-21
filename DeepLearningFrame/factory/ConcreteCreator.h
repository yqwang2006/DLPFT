#pragma once

namespace dlpft{
	namespace factory{
		// 具体工厂类  
		// Product为抽象产品，ConcreteProduct为具体产品  
		template<typename Product, typename ConcreteProduct>  
		class ConcreteCreator  
		{  
		public:  
			// 生产一个具体产品  
			static Product* createProduct()  
			{  
				return new ConcreteProduct();  
			}  
		};  
	};
};
