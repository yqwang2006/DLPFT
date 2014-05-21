#pragma once
#include <map>
#include "ConcreteCreator.h"
namespace dlpft{
	namespace factory{
		// 抽象工厂  
		// Product为抽象产品  
		template<typename Product>  
		class Creator  
		{  
			// 单例实现  
		public:  
			static Creator& Instance()  
			{  
				static Creator<Product> instance;  
				return instance;  
			}  
		private:  
			Creator() {}  
			~Creator() {}  
			Creator(Creator&);  
			// 对外接口  
		public:  
			typedef Product* (*CreateProductDelegate)( ); // 生产产品的函数委托  
			typedef std::map<std::string, CreateProductDelegate> MapRegisterCreatorItem;  
			// 根据具体产品生成具体工厂  
			// 并将其注册进抽象工厂  
			// ConcreteProduct为具体产品  
			template<typename ConcreteProduct>  
			void registerCreator(const std::string& _type)  
			{  
				mConcreteCreators[_type] = ConcreteCreator<Product, ConcreteProduct>::createProduct;  
			}  
			// 删除所有具体工厂  
			void unregisterAllCreators()  
			{  
				mConcreteCreators.clear();  
			}  
			// 生产类型为_type的产品  
			// 失败返回0  
			Product* createProduct(const std::string& _type)  
			{  
				MapRegisterCreatorItem::iterator type = mConcreteCreators.find(_type);  
				if (type != mConcreteCreators.end())  
				{  
					CreateProductDelegate create = type->second;  
					if (create != 0)  
						return create();  
				}  
				return 0;  
			}  
		private:  
			MapRegisterCreatorItem mConcreteCreators; // 保存所有注册过的具体工厂  
		};  
	};
};